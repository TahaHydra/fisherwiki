package com.fisherwiki.core.db

import java.io.Closeable

/**
 * Minimal read-only SQLite surface, so `:core` can query a pack's species
 * database without importing either JDBC or `android.database`.
 *
 * This is the one place where the desktop and the phone genuinely differ:
 * Android ships its own SQLite and `org.xerial:sqlite-jdbc` bundles desktop
 * native libraries, and neither is a good fit for the other. Everything above
 * this interface - all the actual queries, joins and result mapping in
 * [SpeciesRepository] - is shared, so the query logic that ships is the query
 * logic the desktop tests exercise.
 *
 * Deliberately read-only. A pack's database is an immutable, hash-verified
 * artefact; the catch log lives in a separate, writable database owned by the
 * app. Keeping those apart means a pack upgrade can never touch user data.
 */
interface SqliteDriver : Closeable {

    /** Run a query and hand each row to [map], collecting the results. */
    fun <T> query(sql: String, args: List<Any?> = emptyList(), map: (Row) -> T): List<T>

    /** Run a query expected to yield at most one row. */
    fun <T> queryOne(sql: String, args: List<Any?> = emptyList(), map: (Row) -> T): T? =
        query(sql, args, map).firstOrNull()

    /** Column accessors for one result row. */
    interface Row {
        fun getString(index: Int): String?
        fun getLong(index: Int): Long?
        fun getInt(index: Int): Int?
        fun getDouble(index: Int): Double?
        fun isNull(index: Int): Boolean
    }
}

/**
 * Thrown when a pack's database does not match what the engine expects.
 *
 * Treated as a pack-integrity failure, not a recoverable error: continuing with
 * a database whose schema we do not understand risks showing the wrong species
 * for a class index, which is exactly the failure this project is built to
 * avoid.
 */
class SpeciesDatabaseException(message: String, cause: Throwable? = null) :
    RuntimeException(message, cause)
