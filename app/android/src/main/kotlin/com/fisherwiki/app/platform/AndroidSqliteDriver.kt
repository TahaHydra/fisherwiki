package com.fisherwiki.app.platform

import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import com.fisherwiki.core.db.SqliteDriver

/**
 * Android implementation of [SqliteDriver], over the platform's own SQLite.
 *
 * Opened `OPEN_READONLY`: a pack's database is an immutable, hash-verified
 * artefact. If the app could write to it, a pack's SHA-256 would stop matching
 * its manifest and re-verification after an upgrade would fail - and worse, a
 * bug could corrupt data whose whole value is that it is exactly what was
 * shipped. User data lives in a separate writable database.
 */
class AndroidSqliteDriver(path: String) : SqliteDriver {

    private val db: SQLiteDatabase = SQLiteDatabase.openDatabase(
        path, null, SQLiteDatabase.OPEN_READONLY
    )

    private class CursorRow(private val c: Cursor) : SqliteDriver.Row {
        override fun getString(index: Int): String? =
            if (c.isNull(index)) null else c.getString(index)

        override fun getLong(index: Int): Long? =
            if (c.isNull(index)) null else c.getLong(index)

        override fun getInt(index: Int): Int? =
            if (c.isNull(index)) null else c.getInt(index)

        override fun getDouble(index: Int): Double? =
            if (c.isNull(index)) null else c.getDouble(index)

        override fun isNull(index: Int): Boolean = c.isNull(index)
    }

    override fun <T> query(
        sql: String,
        args: List<Any?>,
        map: (SqliteDriver.Row) -> T,
    ): List<T> {
        // rawQuery binds everything as a string. That is fine for our queries -
        // SQLite applies column affinity on comparison - and it keeps the
        // binding path identical to the JDBC driver's, so the desktop tests
        // exercise the same behaviour.
        val bound = args.map { it?.toString() }.toTypedArray()
        db.rawQuery(sql, bound).use { c ->
            val out = ArrayList<T>(c.count.coerceAtMost(1024))
            val row = CursorRow(c)
            while (c.moveToNext()) out.add(map(row))
            return out
        }
    }

    override fun close() {
        runCatching { db.close() }
    }
}
