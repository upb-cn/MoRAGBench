#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cerrno>
#include <cstring>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/index_io.h>
#include <faiss/AutoTune.h>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "FAISS", __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, "FAISS", __VA_ARGS__)

// globals
static faiss::Index* g_index = nullptr;            // may point to IndexIDMap or base index
static faiss::IndexIDMap* g_idmap = nullptr;       // convenience pointer if g_index is an IndexIDMap
static int g_dim = 0;
static int g_ivf_nprobe = 1;

static int g_hnsw_M = 1;
static int g_hnsw_ef_construction = 1;
static int g_hnsw_ef_search = 1;

// next id to assign (64-bit idx_t)
static faiss::idx_t g_next_id = 0;

// Helper: delete current index safely
static void delete_index() {
    if (g_index) {
        try {
            delete g_index;
        } catch (...) {
            // ignore destructor exceptions
        }
        g_index = nullptr;
    }
    g_idmap = nullptr;
    g_dim = 0;
    g_ivf_nprobe = 1;
    g_hnsw_M = 1;
    g_hnsw_ef_construction = 1;
    g_hnsw_ef_search = 1;
    g_next_id = 1;
}

// Helper: compute next id from id_map vector (if IndexIDMap present)
static void recompute_next_id_from_idmap() {
    if (!g_idmap) {
        // mark invalid and log error.
        LOGE("recompute_next_id_from_idmap: IndexIDMap not present — cannot recompute next id");
        g_next_id = (faiss::idx_t)-1;
        return;
    }
    const std::vector<faiss::idx_t> &ids = g_idmap->id_map;
    if (ids.empty()) {
        g_next_id = 0;
        return;
    }
    faiss::idx_t maxid = ids[0];
    for (size_t i = 1; i < ids.size(); ++i) if (ids[i] > maxid) maxid = ids[i];
    g_next_id = maxid + 1;
}

// Helper: wrap a base index into IndexIDMap if not already
static void ensure_idmap_wrapper() {
    if (!g_index) return;
    // If it's already IndexIDMap, set g_idmap
    faiss::IndexIDMap* idmap = dynamic_cast<faiss::IndexIDMap*>(g_index);
    if (idmap) {
        g_idmap = idmap;
        return;
    }
    // Otherwise create an IndexIDMap wrapping the existing index.
    // IndexIDMap takes ownership of the base index pointer.
    g_idmap = new faiss::IndexIDMap(g_index);
    g_index = g_idmap;
}

extern "C" {

/*
 * public native void initFlatNative(int dim);
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_initFlatNative(JNIEnv* env, jobject thiz, jint dim, jint metric) {
    delete_index();
    g_dim = dim;

    faiss::MetricType mt = faiss::METRIC_L2;
    if (metric == 1) mt = faiss::METRIC_INNER_PRODUCT;

    faiss::IndexFlat* flat = nullptr;
    // Create empty string
    std::string indexType;
    if (mt == faiss::METRIC_L2) {
        flat = new faiss::IndexFlatL2(dim);
        indexType = "IndexFlatL2";
    } else {
        flat = new faiss::IndexFlatIP(dim);
        indexType = "IndexFlatIP";
    }

    // Wrap in IndexIDMap so we control ids
    g_idmap = new faiss::IndexIDMap(flat);
    g_index = g_idmap;
    g_next_id = 0;
    // LOGI("Initialized %s with dim=%d", indexType.c_str(), dim);
}

/*
 * public native void initIvfNative(int dim, int nlist, int nprobe, int metric);
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_initIvfNative(JNIEnv* env, jobject thiz, jint dim,
                                                jint nlist, jint nprobe, jint metric) {
    delete_index();
    g_dim = dim;
    g_ivf_nprobe = (int)nprobe;

    faiss::MetricType mt = faiss::METRIC_L2;
    if (metric == 1) mt = faiss::METRIC_INNER_PRODUCT;

    faiss::IndexFlat* quantizer = nullptr;
    if (mt == faiss::METRIC_L2) {
        quantizer = new faiss::IndexFlatL2(dim);
    } else {
        quantizer = new faiss::IndexFlatIP(dim);
    }

    faiss::IndexIVFFlat* ivf = new faiss::IndexIVFFlat(quantizer, dim, (int)nlist, mt);
    ivf->nprobe = g_ivf_nprobe;

    // Wrap in IDMap to control ids and preserve them when serializing
    g_idmap = new faiss::IndexIDMap(ivf);
    g_index = g_idmap;

    g_next_id = 0;
    // LOGI("Initialized IndexIVFFlat dim=%d nlist=%d nprobe=%d metric=%d", dim, nlist, nprobe, metric);
}

/*
 * public native void initHnswNative(int dim, int M, int efConstruction, int efSearch, int metric);
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_initHnswNative(JNIEnv* env, jobject thiz, jint dim,
                                                 jint M, jint efConstruction, jint efSearch, jint metric) {
    delete_index();
    g_dim = dim;
    g_hnsw_M = (int)M;
    g_hnsw_ef_construction = (int)efConstruction;
    g_hnsw_ef_search = (int)efSearch;

    faiss::MetricType mt = faiss::METRIC_L2;
    if (metric == 1) mt = faiss::METRIC_INNER_PRODUCT;

    try {
        faiss::IndexHNSWFlat* hnsw = new faiss::IndexHNSWFlat(g_dim, g_hnsw_M, mt);
        hnsw->hnsw.efConstruction = g_hnsw_ef_construction;
        hnsw->hnsw.efSearch = g_hnsw_ef_search;

        // wrap in idmap
        g_idmap = new faiss::IndexIDMap(hnsw);
        g_index = g_idmap;
        g_next_id = 0;
        /* LOGI("Initialized IndexHNSWFlat dim=%d M=%d efConstruction=%d efSearch=%d metric=%d",
             dim, M, efConstruction, efSearch, metric); */
    } catch (const std::exception &e) {
        LOGE("Exception creating HNSW index: %s", e.what());
    }
}

/*
 * public native void trainIvfNative(float[] trainingVectors, int numVectors);
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_trainIvfNative(JNIEnv* env, jobject thiz, jfloatArray jtrain, jint numVectors) {
    if (!g_index) {
        LOGE("trainIvfNative called but index is null");
        return;
    }
    faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>( (g_idmap ? g_idmap->index : g_index) );
    if (!ivf) {
        LOGE("trainIvfNative called but current index is not an IVF index");
        return;
    }
    if (g_dim <= 0) {
        LOGE("Invalid dimension (%d) for training", g_dim);
        return;
    }

    jsize len = env->GetArrayLength(jtrain);
    jfloat* data = env->GetFloatArrayElements(jtrain, nullptr);

    if (len < numVectors * g_dim) {
        LOGE("trainIvfNative: provided training array length %d is less than numVectors*dim (%d)", len, numVectors * g_dim);
        env->ReleaseFloatArrayElements(jtrain, data, 0);
        return;
    }

    try {
        ivf->train((size_t)numVectors, (const float*)data);
        // LOGI("IVF trained with %d vectors (dim=%d)", numVectors, g_dim);
    } catch (const std::exception &e) {
        LOGE("Exception during ivf->train: %s", e.what());
    }

    env->ReleaseFloatArrayElements(jtrain, data, 0);
}

JNIEXPORT jlongArray JNICALL
Java_com_example_faiss_FaissIndex_addNative(
        JNIEnv* env,
        jobject thiz,
        jfloatArray jvecs,
        jint batchSize) {

    if (!g_idmap || g_dim <= 0) {
        return nullptr;
    }

    const jsize totalLen = env->GetArrayLength(jvecs);
    if (totalLen < batchSize * g_dim) {
        LOGE("addNative: array too small");
        return nullptr;
    }

    jfloat* data = env->GetFloatArrayElements(jvecs, nullptr);

    // Prepare ID array
    std::vector<faiss::idx_t> ids(batchSize);
    for (int i = 0; i < batchSize; i++) {
        ids[i] = g_next_id++;
    }

    try {
        g_idmap->add_with_ids(
                batchSize,
                (const float*)data,
                ids.data()
        );
    } catch (const std::exception &e) {
        LOGE("Exception in addNative: %s", e.what());
        env->ReleaseFloatArrayElements(jvecs, data, 0);
        return nullptr;
    }

    env->ReleaseFloatArrayElements(jvecs, data, 0);

    // Convert ids to jlongArray
    jlongArray result = env->NewLongArray(batchSize);
    std::vector<jlong> out(batchSize);
    for (int i = 0; i < batchSize; i++) {
        out[i] = (jlong)ids[i];
    }
    env->SetLongArrayRegion(result, 0, batchSize, out.data());

    return result;
}

JNIEXPORT jobject JNICALL
Java_com_example_faiss_FaissIndex_queryNative(
        JNIEnv* env,
        jobject thiz,
        jfloatArray jqueries,
        jint queryCount,
        jint k) {

    if (!g_index) {
        LOGE("queryNative called but index is null");
        return nullptr;
    }

    const jsize totalLen = env->GetArrayLength(jqueries);
    if (totalLen < queryCount * g_dim) {
        LOGE("queryNative: array too small");
        return nullptr;
    }

    jfloat* qptr = env->GetFloatArrayElements(jqueries, nullptr);

    // Prepare result buffers (queryCount × k)
    std::vector<float> distances(queryCount * k);
    std::vector<faiss::idx_t> labels(queryCount * k);

    // Apply IVF/HNSW params exactly like in single query
    faiss::Index* baseIndex = g_idmap ? g_idmap->index : g_index;

    faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>(baseIndex);
    if (ivf) {
        ivf->nprobe = g_ivf_nprobe;
    }

    faiss::IndexHNSWFlat* hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(baseIndex);
    if (hnsw) {
        hnsw->hnsw.efSearch = g_hnsw_ef_search;
    }

    try {
        baseIndex->search(
                queryCount,
                (const float*)qptr,
                k,
                distances.data(),
                labels.data()
        );
    } catch (const std::exception &e) {
        LOGE("Exception during search: %s", e.what());
        env->ReleaseFloatArrayElements(jqueries, qptr, 0);
        return nullptr;
    }

    env->ReleaseFloatArrayElements(jqueries, qptr, 0);

    // Convert labels to jlong[]
    jlongArray jlabels = env->NewLongArray(queryCount * k);
    std::vector<jlong> labels_j(queryCount * k);
    for (int i = 0; i < queryCount * k; ++i) {
        labels_j[i] = (jlong)labels[i];
    }
    env->SetLongArrayRegion(jlabels, 0, labels_j.size(), labels_j.data());

    // Convert distances to jfloat[]
    jfloatArray jdistances = env->NewFloatArray(queryCount * k);
    env->SetFloatArrayRegion(jdistances, 0, distances.size(), distances.data());

    // Build same QueryResults object
    jclass resultClass = env->FindClass("com/example/faiss/FaissIndex$QueryResults");
    if (!resultClass) {
        LOGE("Could not find QueryResults class");
        return nullptr;
    }

    jmethodID ctor = env->GetMethodID(resultClass, "<init>", "([J[F)V");
    if (!ctor) {
        LOGE("Could not find QueryResults constructor");
        return nullptr;
    }

    jobject resultObj = env->NewObject(resultClass, ctor, jlabels, jdistances);
    return resultObj;
}


/*
 * public native float[] getVectorNative(long id);
 */
JNIEXPORT jfloatArray JNICALL
Java_com_example_faiss_FaissIndex_getVectorNative(JNIEnv* env, jobject thiz, jlong id) {
    if (!g_index) {
        LOGE("getVectorNative called but index is null");
        return nullptr;
    }
    if (g_dim <= 0) {
        LOGE("getVectorNative: invalid dim=%d", g_dim);
        return nullptr;
    }
    std::vector<float> vec(g_dim);
    try {
        g_index->reconstruct((faiss::idx_t)id, vec.data());
    } catch (const std::exception &e) {
        LOGE("Exception during reconstruct: %s", e.what());
        return nullptr;
    }
    jfloatArray out = env->NewFloatArray(g_dim);
    env->SetFloatArrayRegion(out, 0, g_dim, vec.data());
    return out;
}

/*
 * public native void setNprobeNative(int nprobe);
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_setNprobeNative(JNIEnv* env, jobject thiz, jint nprobe) {
    g_ivf_nprobe = (int)nprobe;
    faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>( (g_idmap ? g_idmap->index : g_index) );
    if (ivf) {
        ivf->nprobe = g_ivf_nprobe;
        // LOGI("setNprobe: updated ivf->nprobe=%d", g_ivf_nprobe);
    } else {
        LOGE("setNprobe: index is not IVF; stored nprobe=%d will be used if IVF created", g_ivf_nprobe);
    }
}

/*
 * public native long getTotalNative();
 */
JNIEXPORT jlong JNICALL
Java_com_example_faiss_FaissIndex_getTotalNative(JNIEnv* env, jobject thiz) {
    if (!g_index) return (jlong)0;
    return (jlong)g_index->ntotal;
}

/*
 * Persist FAISS index to path (string)
 * Java signature: public native void writeIndexNative(String path);
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_writeIndexNative(JNIEnv* env, jobject thiz, jstring jpath) {

    if (!g_index) {
        LOGE("writeIndexNative: no index to write");
        // Throw to make Java handle fatal error
        jclass exc = env->FindClass("java/lang/IllegalStateException");
        env->ThrowNew(exc, "writeIndexNative: no index to write");
        return;
    }

    // Require that the current index is an IndexIDMap.
    faiss::IndexIDMap* idmap = dynamic_cast<faiss::IndexIDMap*>(g_index);
    if (!idmap) {
        LOGE("writeIndexNative: current index is not an IndexIDMap; refusing to write in strict-id mode");
        jclass exc = env->FindClass("java/lang/IllegalStateException");
        env->ThrowNew(exc, "writeIndexNative: index does not preserve IDs (not IndexIDMap)");
        return;
    }

    const char* path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) {
        LOGE("writeIndexNative: GetStringUTFChars returned null");
        jclass exc = env->FindClass("java/lang/IllegalArgumentException");
        env->ThrowNew(exc, "writeIndexNative: null path");
        return;
    }

    std::string pathStr(path);
    std::string tmp = pathStr + ".tmp";
    std::string metaf = pathStr + ".meta";

    try {
        // write atomic: write to tmp then rename
        faiss::write_index(g_index, tmp.c_str());
        // POSIX rename will atomically replace
        // move tmp -> final
        if (std::rename(tmp.c_str(), pathStr.c_str()) != 0) {
            // cleanup tmp if exists
            std::remove(tmp.c_str());
            env->ReleaseStringUTFChars(jpath, path);
            std::string msg = std::string("writeIndexNative: failed to rename temp file to final location: ") + std::strerror(errno);
            jclass exc = env->FindClass("java/io/IOException");
            env->ThrowNew(exc, msg.c_str());
            return;
        }

        // Write metadata file next to the index (simple key=value text). If this fails -> treat as error.
        std::ofstream of(metaf.c_str(), std::ios::out | std::ios::trunc);
        if (!of.is_open()) {
            // Remove the written index since metadata couldn't be saved (strict failure mode requested)
            std::remove(pathStr.c_str());
            env->ReleaseStringUTFChars(jpath, path);
            std::string msg = std::string("writeIndexNative: could not open meta file for writing: ") + metaf;
            jclass exc = env->FindClass("java/io/IOException");
            env->ThrowNew(exc, msg.c_str());
            return;
        }

        // write values
        of << "g_dim=" << g_dim << "\n";
        of << "g_ivf_nprobe=" << g_ivf_nprobe << "\n";
        of << "g_hnsw_M=" << g_hnsw_M << "\n";
        of << "g_hnsw_ef_construction=" << g_hnsw_ef_construction << "\n";
        of << "g_hnsw_ef_search=" << g_hnsw_ef_search << "\n";

    } catch (const std::exception &e) {
        LOGE("Exception writing index: %s", e.what());
        env->ReleaseStringUTFChars(jpath, path);
        jclass exc = env->FindClass("java/io/IOException");
        env->ThrowNew(exc, std::string("writeIndexNative: error writing temp index: " + std::string(e.what())).c_str());
        return;
    }

    env->ReleaseStringUTFChars(jpath, path);
}

/*
 * Load FAISS index from path (string)
 * Java signature: public native boolean readIndexNative(String path);
 * returns true on success
 *
 * Behaviour:
 *  - load index
 *  - require the loaded index to be an IndexIDMap (fail otherwise)
 *  - replace existing index only on success; otherwise delete loaded index and return false
 */
JNIEXPORT jboolean JNICALL
Java_com_example_faiss_FaissIndex_readIndexNative(JNIEnv* env, jobject thiz, jstring jpath) {
    const char* path = env->GetStringUTFChars(jpath, nullptr);
    if (!path) {
        LOGE("readIndexNative: null path");
        jclass exc = env->FindClass("java/lang/IllegalArgumentException");
        env->ThrowNew(exc, "readIndexNative: null path");
        return JNI_FALSE;
    }

    std::string pathStr(path);
    std::string metaf = pathStr + ".meta";

    faiss::Index* new_index = nullptr;
    try {
        new_index = faiss::read_index(pathStr.c_str());
    } catch (const std::exception &e) {
        LOGE("Exception reading index %s: %s", path, e.what());
        env->ReleaseStringUTFChars(jpath, path);
        jclass exc = env->FindClass("java/io/IOException");
        env->ThrowNew(exc, std::string("readIndexNative: error reading index: " + std::string(e.what())).c_str());
        return JNI_FALSE;
    }

    // Check that the loaded index is an IndexIDMap (strict-id requirement)
    faiss::IndexIDMap* loaded_idmap = dynamic_cast<faiss::IndexIDMap*>(new_index);
    if (!loaded_idmap) {
        LOGE("readIndexNative: loaded index is not an IndexIDMap; refusing to load in strict-id mode");
        // Clean up the index we just loaded to avoid leaks
        try { delete new_index; } catch (...) {}
        env->ReleaseStringUTFChars(jpath, path);
        jclass exc = env->FindClass("java/lang/IllegalStateException");
        env->ThrowNew(exc, "readIndexNative: loaded index does not preserve IDs (not IndexIDMap)");
        return JNI_FALSE;
    }

    // Replace current index with the loaded IndexIDMap
    delete_index(); // deletes existing g_index safely
    g_index = loaded_idmap; // ownership transferred
    g_idmap = loaded_idmap;

    // Open meta file (strict: must exist)
    std::ifstream inf(metaf.c_str());
    if (!inf.is_open()) {
        // cleanup loaded index
        try { delete new_index; } catch (...) {}
        env->ReleaseStringUTFChars(jpath, path);
        std::string msg = std::string("readIndexNative: failed to open meta file: ") + metaf;
        jclass exc = env->FindClass("java/io/IOException");
        env->ThrowNew(exc, msg.c_str());
        return JNI_FALSE;
    }

    // parse meta; require keys: g_ivf_nprobe, g_hnsw_M, g_hnsw_ef_construction, g_hnsw_ef_search, index_type
    bool has_ivf_nprobe = false, has_hnsw_M = false, has_hnsw_ef_construction = false, has_hnsw_ef_search = false;

    std::string line;
    while (std::getline(inf, line)) {
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        try {
            if (key == "g_dim") {
                g_dim = std::stoi(val);
            } else if (key == "g_ivf_nprobe") {
                g_ivf_nprobe = std::stoi(val);
                has_ivf_nprobe = true;
            } else if (key == "g_hnsw_M") {
                g_hnsw_M = std::stoi(val);
                has_hnsw_M = true;
            } else if (key == "g_hnsw_ef_construction") {
                g_hnsw_ef_construction = std::stoi(val);
                has_hnsw_ef_construction = true;
            } else if (key == "g_hnsw_ef_search") {
                g_hnsw_ef_search = std::stoi(val);
                has_hnsw_ef_search = true;
            }
        } catch (...) {
            // parse error for a value: treat as fatal
            inf.close();
            try { delete new_index; } catch (...) {}
            env->ReleaseStringUTFChars(jpath, path);
            jclass exc = env->FindClass("java/io/IOException");
            env->ThrowNew(exc, std::string("readIndexNative: parse error in meta file for key " + key).c_str());
            return JNI_FALSE;
        }
    }
    inf.close();

    // If index is ivf, ensure ivf_nprobe present. If hnsw, ensure hnsw keys present. For strictness, require all four keys always:
    if (!has_ivf_nprobe || !has_hnsw_M || !has_hnsw_ef_construction || !has_hnsw_ef_search) {
        try { delete new_index; } catch (...) {}
        env->ReleaseStringUTFChars(jpath, path);
        jclass exc = env->FindClass("java/io/IOException");
        env->ThrowNew(exc, "readIndexNative: meta file missing one or more required parameters (g_ivf_nprobe, g_hnsw_M, g_hnsw_ef_construction, g_hnsw_ef_search)");
        return JNI_FALSE;
    }

    // apply parameters depending on loaded index type (and validate index_type matches actual loaded type)
    try {
        if (auto ivf = dynamic_cast<faiss::IndexIVF*>(g_idmap->index)) {
            g_dim = ivf->d;
            ivf->nprobe = g_ivf_nprobe;
        } else if (auto hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(g_idmap->index)) {
            g_dim = hnsw->d;
            // set efConstruction/efSearch if available
            hnsw->hnsw.efConstruction = g_hnsw_ef_construction;
            hnsw->hnsw.efSearch = g_hnsw_ef_search;
            // Note: M cannot be changed after construction — we keep g_hnsw_M updated though
        } else if (auto flat = dynamic_cast<faiss::IndexFlat*>(g_idmap->index)) {
            g_dim = flat->d;
        } else {
            // unknown index type
            delete_index();
            env->ReleaseStringUTFChars(jpath, path);
            jclass exc = env->FindClass("java/io/IOException");
            env->ThrowNew(exc, "readIndexNative: unknown index_type value in meta");
            return JNI_FALSE;
        }
    } catch (const std::exception &e) {
        // unexpected runtime error
        delete_index();
        env->ReleaseStringUTFChars(jpath, path);
        jclass exc = env->FindClass("java/io/IOException");
        env->ThrowNew(exc, std::string("readIndexNative: error applying metadata: " + std::string(e.what())).c_str());
        return JNI_FALSE;
    }

    // Recompute next id strictly from id_map
    recompute_next_id_from_idmap();
    if (g_next_id == (faiss::idx_t)-1) {
        // recompute failed (shouldn't happen since we confirmed idmap), but guard anyway
        LOGE("readIndexNative: failed to recompute next id from IndexIDMap");
        env->ReleaseStringUTFChars(jpath, path);
        jclass exc = env->FindClass("java/lang/IllegalStateException");
        env->ThrowNew(exc, "readIndexNative: failed to recompute next id");
        return JNI_FALSE;
    }

    /* LOGI("Read faiss index from %s, ntotal=%lld, next_id=%lld, dim=%d",
         path, (long long)g_index->ntotal, (long long)g_next_id, g_dim); */

    env->ReleaseStringUTFChars(jpath, path);
    return JNI_TRUE;
}

// Returns a diagnostic string for IVF indexes (or general stats if not IVF)
JNIEXPORT jstring JNICALL
Java_com_example_faiss_FaissIndex_getIVFDebugNative(JNIEnv* env, jobject thiz) {
    if (!g_index) {
        return env->NewStringUTF("no index loaded");
    }

    std::ostringstream oss;
    oss << "ntotal=" << g_index->ntotal;

    // try IndexIDMap wrapper
    faiss::Index* core = nullptr;
    if (g_idmap) core = g_idmap->index; else core = g_index;

    if (auto ivf = dynamic_cast<faiss::IndexIVF*>(core)) {
        oss << " type=IVF";
        oss << " d=" << ivf->d;
        oss << " nlist=" << ivf->nlist;
        oss << " nprobe=" << ivf->nprobe;
        // count non-empty lists (safe API: list_size())
        long nonempty = 0;
        for (int i = 0; i < ivf->nlist; ++i) {
            if (ivf->invlists->list_size(i) > 0) ++nonempty;
        }
        oss << " nonempty_lists=" << nonempty;
    } else if (auto h = dynamic_cast<faiss::IndexHNSWFlat*>(core)) {
        oss << " type=HNSW";
        oss << " d=" << h->d;
        oss << " M=" << g_hnsw_M;
        oss << " efSearch=" << h->hnsw.efSearch;
        oss << " efConstruction=" << h->hnsw.efConstruction;
    } else if (auto f = dynamic_cast<faiss::IndexFlat*>(core)) {
        oss << " type=FLAT d=" << f->d;
    } else {
        oss << " type=unknown";
    }

    std::string s = oss.str();
    return env->NewStringUTF(s.c_str());
}

/*
 * reset/get stats and clear index implementations
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_resetFaissStatsNative(JNIEnv* env, jobject thiz) {
    try {
#ifdef FAISS_HAS_INDEX_IVF_STATS
        faiss::indexIVF_stats.reset();
#endif
#ifdef FAISS_HAS_HNSW_STATS
        faiss::hnsw_stats.reset();
#endif
    } catch (...) {}
}

JNIEXPORT jstring JNICALL
Java_com_example_faiss_FaissIndex_getIvfStatsNative(JNIEnv* env, jobject thiz) {
    std::string json = "{}";
    try {
        faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>((g_idmap ? g_idmap->index : g_index));
        std::ostringstream oss;
        if (ivf) {
            oss << "{";
            oss << "\"ntotal\":" << g_index->ntotal << ",";
#ifdef FAISS_HAS_INDEX_IVF_STATS
            oss << "\"ivf_nq\":" << faiss::indexIVF_stats.nq << ",";
            oss << "\"ivf_nlist\":" << faiss::indexIVF_stats.nlist << ",";
            oss << "\"ivf_ndis\":" << faiss::indexIVF_stats.ndis << ",";
            oss << "\"ivf_search_time_ms\":" << faiss::indexIVF_stats.search_time << ",";
            oss << "\"ivf_quant_time_ms\":" << faiss::indexIVF_stats.quantization_time;
#else
            oss << "\"note\":\"ivf-stats-not-available\"";
#endif
            oss << "}";
            json = oss.str();
        } else {
            oss << "{\"type\":\"not-ivf\",\"ntotal\":" << (g_index ? g_index->ntotal : 0) << "}";
            json = oss.str();
        }
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "{\"error\":\"" << e.what() << "\"}";
        json = oss.str();
    }
    return env->NewStringUTF(json.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_example_faiss_FaissIndex_getHnswStatsNative(JNIEnv* env, jobject thiz) {
    std::string json = "{}";
    try {
        faiss::IndexHNSWFlat* hnsw = dynamic_cast<faiss::IndexHNSWFlat*>((g_idmap ? dynamic_cast<faiss::IndexHNSWFlat*>(g_idmap->index) : dynamic_cast<faiss::IndexHNSWFlat*>(g_index)));
        std::ostringstream oss;
        if (hnsw) {
            oss << "{";
            oss << "\"ntotal\":" << g_index->ntotal << ",";
#ifdef FAISS_HAS_HNSW_STATS
            oss << "\"nhops\":" << faiss::hnsw_stats.nhops << ",";
            oss << "\"n1\":" << faiss::hnsw_stats.n1 << ",";
            oss << "\"n2\":" << faiss::hnsw_stats.n2 << ",";
            oss << "\"ndis\":" << faiss::hnsw_stats.ndis;
#else
            oss << "\"note\":\"hnsw-stats-not-available\"";
#endif
            oss << "}";
            json = oss.str();
        } else {
            oss << "{\"type\":\"not-hnsw\",\"ntotal\":" << (g_index ? g_index->ntotal : 0) << "}";
            json = oss.str();
        }
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "{\"error\":\"" << e.what() << "\"}";
        json = oss.str();
    }
    return env->NewStringUTF(json.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_example_faiss_FaissIndex_getFlatStatsNative(JNIEnv* env, jobject thiz) {
    std::ostringstream oss;
    if (!g_index) {
        oss << "{\"error\":\"no-index\"}";
        return env->NewStringUTF(oss.str().c_str());
    }
    oss << "{";
    oss << "\"ntotal\":" << g_index->ntotal << ",";
    oss << "\"distances_per_query\":" << g_index->ntotal;
    oss << "}";
    return env->NewStringUTF(oss.str().c_str());
}

JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_clearNative(JNIEnv* env, jobject thiz) {
    delete_index();
    LOGI("Cleared FAISS index");
}

} // extern "C"
