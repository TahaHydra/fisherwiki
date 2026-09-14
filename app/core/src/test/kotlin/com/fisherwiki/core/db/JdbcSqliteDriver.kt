package com.fisherwiki.core.db

import java.sql.Connection
import java.sql.DriverManager
import java.sql.ResultSet

/**
 * Desktop [SqliteDriver] backed by `org.xerial:sqlite-jdbc`.
 *
 * Test-scope only. Its existence is what lets the whole of
 * [SpeciesRepository] - every query, join and result mapping that ships - be
 * exercised against a real pack database on a workstation, with the Android
 * driver providing the same three methods on device.
 */
class JdbcSqliteDriver(path: String, readOnly: Boolean = true) : SqliteDriver {

    private val conn: Connection = DriverManager.getConnection(
        "jdbc:sqlite:$path" + if (readOnly) "?open_mode=1" else ""
    )

    private class JdbcRow(private val rs: ResultSet) : SqliteDriver.Row {
        // JDBC columns are 1-based; the interface is 0-based, like Android's.
        override fun getString(index: Int): String? = rs.getString(index + 1)
        override fun getLong(index: Int): Long? =
            rs.getLong(index + 1).takeUnless { rs.wasNull() }
        override fun getInt(index: Int): Int? =
            rs.getInt(index + 1).takeUnless { rs.wasNull() }
        override fun getDouble(index: Int): Double? =
            rs.getDouble(index + 1).takeUnless { rs.wasNull() }
        override fun isNull(index: Int): Boolean {
            rs.getObject(index + 1)
            return rs.wasNull()
        }
    }

    override fun <T> query(
        sql: String,
        args: List<Any?>,
        map: (SqliteDriver.Row) -> T,
    ): List<T> {
        conn.prepareStatement(sql).use { st ->
            args.forEachIndexed { i, a ->
                when (a) {
                    null -> st.setObject(i + 1, null)
                    is Int -> st.setInt(i + 1, a)
                    is Long -> st.setLong(i + 1, a)
                    is Double -> st.setDouble(i + 1, a)
                    is Float -> st.setDouble(i + 1, a.toDouble())
                    is Boolean -> st.setInt(i + 1, if (a) 1 else 0)
                    else -> st.setString(i + 1, a.toString())
                }
            }
            st.executeQuery().use { rs ->
                val out = ArrayList<T>()
                val row = JdbcRow(rs)
                while (rs.next()) out.add(map(row))
                return out
            }
        }
    }

    override fun close() {
        runCatching { conn.close() }
    }
}
