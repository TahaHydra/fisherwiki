package com.fisherwiki.app.data

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import com.fisherwiki.core.model.Identification
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * The user's local catch log.
 *
 * Privacy contract, and it is the reason this is a separate database from the
 * pack's:
 *
 * * it lives in the app's private storage and is **never** uploaded;
 * * there is no sync code, no analytics hook and no network call anywhere in
 *   this file or anything it touches;
 * * coordinates are optional per catch, stored only if the user chose to attach
 *   them, and can be stripped on export;
 * * a pack upgrade replaces the pack database and cannot touch this one;
 * * [exportJson] exists so the data is *portable*, not so it is collectable -
 *   the user initiates it and chooses where it goes.
 *
 * Corrections the user makes to an identification are kept
 * ([CatchRecord.correctedTaxonId]) because they are the single most valuable
 * signal for improving the model. They are still never sent anywhere
 * automatically; `docs/PRIVACY.md` describes the opt-in contribution flow.
 */
class CatchLog(context: Context) {

    private val helper = Helper(context.applicationContext)

    private class Helper(context: Context) :
        SQLiteOpenHelper(context, DB_NAME, null, DB_VERSION) {

        override fun onCreate(db: SQLiteDatabase) {
            db.execSQL(
                """
                CREATE TABLE catches (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at        INTEGER NOT NULL,
                    photo_paths       TEXT NOT NULL,   -- JSON array, app-private
                    identified_taxon_id INTEGER,
                    identified_name   TEXT,
                    corrected_taxon_id INTEGER,
                    corrected_name    TEXT,
                    certainty         TEXT,
                    confidence        REAL,
                    alternatives_json TEXT,
                    latitude          REAL,
                    longitude         REAL,
                    location_accuracy REAL,
                    length_mm         INTEGER,
                    weight_g          INTEGER,
                    notes             TEXT,
                    released          INTEGER,
                    model_version     TEXT,
                    pack_id           TEXT,
                    pack_version      INTEGER,
                    inference_ms      INTEGER
                )
                """.trimIndent()
            )
            db.execSQL("CREATE INDEX idx_catches_created ON catches(created_at DESC)")
            db.execSQL("CREATE INDEX idx_catches_taxon ON catches(identified_taxon_id)")
        }

        override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            // Migrations are additive only. A user's catch log is irreplaceable
            // - they cannot re-catch the fish - so no migration may drop a
            // column or a row. Future versions add columns with defaults.
            if (oldVersion < 2 && newVersion >= 2) {
                // (placeholder for the first real migration)
            }
        }

        override fun onDowngrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
            // Default behaviour deletes the database. Never do that here.
            throw IllegalStateException(
                "Refusing to downgrade the catch log from $oldVersion to $newVersion; " +
                    "that would destroy user data."
            )
        }

        companion object {
            const val DB_NAME = "catch_log.db"
            const val DB_VERSION = 1
        }
    }

    data class CatchRecord(
        val id: Long = 0,
        val createdAt: Long = System.currentTimeMillis(),
        val photoPaths: List<String> = emptyList(),
        val identifiedTaxonId: Long? = null,
        val identifiedName: String? = null,
        val correctedTaxonId: Long? = null,
        val correctedName: String? = null,
        val certainty: String? = null,
        val confidence: Double? = null,
        val alternatives: List<Pair<String, Double>> = emptyList(),
        val latitude: Double? = null,
        val longitude: Double? = null,
        val locationAccuracy: Double? = null,
        val lengthMm: Int? = null,
        val weightG: Int? = null,
        val notes: String? = null,
        val released: Boolean? = null,
        val modelVersion: String? = null,
        val packId: String? = null,
        val packVersion: Int? = null,
        val inferenceMs: Long? = null,
    ) {
        /** The user's correction wins over the model's guess for display. */
        val displayName: String?
            get() = correctedName ?: identifiedName

        val wasCorrected: Boolean get() = correctedTaxonId != null
    }

    fun insert(record: CatchRecord): Long {
        val v = ContentValues().apply {
            put("created_at", record.createdAt)
            put("photo_paths", JSONArray(record.photoPaths).toString())
            record.identifiedTaxonId?.let { put("identified_taxon_id", it) }
            put("identified_name", record.identifiedName)
            record.correctedTaxonId?.let { put("corrected_taxon_id", it) }
            put("corrected_name", record.correctedName)
            put("certainty", record.certainty)
            record.confidence?.let { put("confidence", it) }
            put("alternatives_json", JSONArray(record.alternatives.map {
                JSONObject().put("name", it.first).put("p", it.second)
            }).toString())
            record.latitude?.let { put("latitude", it) }
            record.longitude?.let { put("longitude", it) }
            record.locationAccuracy?.let { put("location_accuracy", it) }
            record.lengthMm?.let { put("length_mm", it) }
            record.weightG?.let { put("weight_g", it) }
            put("notes", record.notes)
            record.released?.let { put("released", if (it) 1 else 0) }
            put("model_version", record.modelVersion)
            put("pack_id", record.packId)
            record.packVersion?.let { put("pack_version", it) }
            record.inferenceMs?.let { put("inference_ms", it) }
        }
        return helper.writableDatabase.insert("catches", null, v)
    }

    /** Record the user's correction. The original prediction is preserved. */
    fun correct(id: Long, taxonId: Long, name: String) {
        val v = ContentValues().apply {
            put("corrected_taxon_id", taxonId)
            put("corrected_name", name)
        }
        helper.writableDatabase.update("catches", v, "id = ?", arrayOf(id.toString()))
    }

    fun update(record: CatchRecord) {
        val v = ContentValues().apply {
            put("length_mm", record.lengthMm)
            put("weight_g", record.weightG)
            put("notes", record.notes)
            record.released?.let { put("released", if (it) 1 else 0) }
        }
        helper.writableDatabase.update(
            "catches", v, "id = ?", arrayOf(record.id.toString())
        )
    }

    fun delete(id: Long, alsoDeletePhotos: Boolean = true) {
        if (alsoDeletePhotos) {
            get(id)?.photoPaths?.forEach { runCatching { File(it).delete() } }
        }
        helper.writableDatabase.delete("catches", "id = ?", arrayOf(id.toString()))
    }

    fun get(id: Long): CatchRecord? =
        query("SELECT * FROM catches WHERE id = ?", arrayOf(id.toString())).firstOrNull()

    fun all(limit: Int = 500, offset: Int = 0): List<CatchRecord> = query(
        "SELECT * FROM catches ORDER BY created_at DESC LIMIT ? OFFSET ?",
        arrayOf(limit.toString(), offset.toString()),
    )

    fun count(): Int =
        helper.readableDatabase.rawQuery("SELECT count(*) FROM catches", null).use {
            if (it.moveToFirst()) it.getInt(0) else 0
        }

    /** Catches the user corrected: the training-feedback set. */
    fun corrections(): List<CatchRecord> =
        query("SELECT * FROM catches WHERE corrected_taxon_id IS NOT NULL " +
            "ORDER BY created_at DESC", emptyArray())

    private fun query(sql: String, args: Array<String>): List<CatchRecord> {
        helper.readableDatabase.rawQuery(sql, args).use { c ->
            val out = ArrayList<CatchRecord>()
            fun str(n: String) = c.getColumnIndex(n).let {
                if (it < 0 || c.isNull(it)) null else c.getString(it)
            }
            fun long(n: String) = c.getColumnIndex(n).let {
                if (it < 0 || c.isNull(it)) null else c.getLong(it)
            }
            fun dbl(n: String) = c.getColumnIndex(n).let {
                if (it < 0 || c.isNull(it)) null else c.getDouble(it)
            }
            fun int(n: String) = c.getColumnIndex(n).let {
                if (it < 0 || c.isNull(it)) null else c.getInt(it)
            }
            while (c.moveToNext()) {
                val paths = runCatching {
                    val arr = JSONArray(str("photo_paths") ?: "[]")
                    (0 until arr.length()).map { arr.getString(it) }
                }.getOrDefault(emptyList())
                val alts = runCatching {
                    val arr = JSONArray(str("alternatives_json") ?: "[]")
                    (0 until arr.length()).map {
                        val o = arr.getJSONObject(it)
                        o.getString("name") to o.getDouble("p")
                    }
                }.getOrDefault(emptyList())
                out.add(
                    CatchRecord(
                        id = long("id") ?: 0,
                        createdAt = long("created_at") ?: 0,
                        photoPaths = paths,
                        identifiedTaxonId = long("identified_taxon_id"),
                        identifiedName = str("identified_name"),
                        correctedTaxonId = long("corrected_taxon_id"),
                        correctedName = str("corrected_name"),
                        certainty = str("certainty"),
                        confidence = dbl("confidence"),
                        alternatives = alts,
                        latitude = dbl("latitude"),
                        longitude = dbl("longitude"),
                        locationAccuracy = dbl("location_accuracy"),
                        lengthMm = int("length_mm"),
                        weightG = int("weight_g"),
                        notes = str("notes"),
                        released = int("released")?.let { it == 1 },
                        modelVersion = str("model_version"),
                        packId = str("pack_id"),
                        packVersion = int("pack_version"),
                        inferenceMs = long("inference_ms"),
                    )
                )
            }
            return out
        }
    }

    /**
     * Export as JSON for the user to keep or share.
     *
     * @param includeLocation when false, coordinates are omitted entirely.
     *   Defaults to **false**: an export is more likely to be shared than a
     *   local database, and a fishing spot is exactly the kind of thing a user
     *   does not intend to publish. The UI asks explicitly.
     * @param onlyCorrections export just the corrections, for the opt-in model
     *   improvement flow.
     */
    fun exportJson(
        includeLocation: Boolean = false,
        onlyCorrections: Boolean = false,
    ): String {
        val records = if (onlyCorrections) corrections() else all(limit = 100_000)
        val arr = JSONArray()
        for (r in records) {
            val o = JSONObject()
            o.put("created_at", r.createdAt)
            o.put("identified_taxon_id", r.identifiedTaxonId ?: JSONObject.NULL)
            o.put("identified_name", r.identifiedName ?: JSONObject.NULL)
            o.put("corrected_taxon_id", r.correctedTaxonId ?: JSONObject.NULL)
            o.put("corrected_name", r.correctedName ?: JSONObject.NULL)
            o.put("certainty", r.certainty ?: JSONObject.NULL)
            o.put("confidence", r.confidence ?: JSONObject.NULL)
            o.put("length_mm", r.lengthMm ?: JSONObject.NULL)
            o.put("weight_g", r.weightG ?: JSONObject.NULL)
            o.put("notes", r.notes ?: JSONObject.NULL)
            o.put("model_version", r.modelVersion ?: JSONObject.NULL)
            o.put("pack_id", r.packId ?: JSONObject.NULL)
            o.put("pack_version", r.packVersion ?: JSONObject.NULL)
            if (includeLocation) {
                o.put("latitude", r.latitude ?: JSONObject.NULL)
                o.put("longitude", r.longitude ?: JSONObject.NULL)
            }
            arr.put(o)
        }
        return JSONObject()
            .put("format", "fisherwiki-catchlog-v1")
            .put("exported_at", System.currentTimeMillis())
            .put("includes_location", includeLocation)
            .put("count", arr.length())
            .put("catches", arr)
            .toString(2)
    }

    fun close() = helper.close()

    companion object {
        /**
         * Build a record from an identification result.
         *
         * @param correction the user's own pick, when they saved via "Not
         *   right?" rather than "Save catch". Recorded as *corrected*, not as
         *   what the model said: [identifiedTaxonId]/[identifiedName] always
         *   preserve the model's original guess (or its coarse fallback),
         *   even when the user disagreed with it from the very first save -
         *   correction data is only useful for improving the model if it is
         *   honest about what the model actually predicted.
         */
        fun from(
            identification: Identification,
            photoPaths: List<String>,
            latitude: Double? = null,
            longitude: Double? = null,
            correction: Pair<Long, String>? = null,
        ): CatchRecord = CatchRecord(
            photoPaths = photoPaths,
            identifiedTaxonId = identification.best?.taxon?.id
                ?: identification.coarseFallback?.taxon?.id,
            identifiedName = identification.best?.taxon?.scientificName
                ?: identification.coarseFallback?.taxon?.scientificName,
            correctedTaxonId = correction?.first,
            correctedName = correction?.second,
            certainty = identification.certainty.name,
            confidence = identification.best?.probability?.toDouble()
                ?: identification.coarseFallback?.probability?.toDouble(),
            alternatives = identification.alternatives.map {
                it.taxon.scientificName to it.probability.toDouble()
            },
            latitude = latitude,
            longitude = longitude,
            packId = identification.packId,
            packVersion = identification.packVersion,
            inferenceMs = identification.inferenceMillis,
        )
    }
}
