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
#include <sched.h>
#include <unistd.h>
#include <new>
#include <faiss/IndexScalarQuantizer.h>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/index_io.h>
#include <faiss/AutoTune.h>
#include <faiss/IndexBinary.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>

// Thread control (symbols come from the prebuilt libomp.a and libopenblas.a)
extern "C" void omp_set_num_threads(int);
extern "C" int omp_get_max_threads(void);
extern "C" void openblas_set_num_threads(int);
extern "C" int openblas_get_num_threads(void);

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

// Number of threads FAISS / OpenBLAS may use. 0 = library default (all cores).
// Re-applied at the start of every train/add/search call, because OpenMP keeps the
// setting per calling thread and JNI calls can arrive on different Java threads.
static int g_num_threads = 0;
// CPU cores the calling thread is pinned to before FAISS work (empty = no pinning).
// sched_setaffinity(0, ...) affects only the calling thread, so it is re-applied on
// every call, like the thread count. Intended for num_threads = 1.
static std::vector<int> g_pin_cores;
// Library defaults, captured on first use, so that "0 threads" / "no pinning" restores
// the original state instead of keeping whatever an earlier config set.
static bool g_defaults_saved = false;
static int g_default_omp_threads = 0;
static int g_default_blas_threads = 0;
static cpu_set_t g_default_affinity;
static void save_thread_defaults() {
    if (g_defaults_saved) return;
    g_default_omp_threads = omp_get_max_threads();
    g_default_blas_threads = openblas_get_num_threads();
    CPU_ZERO(&g_default_affinity);
    sched_getaffinity(0, sizeof(g_default_affinity), &g_default_affinity);
    g_defaults_saved = true;
}
static void apply_num_threads() {
    if (g_num_threads > 0) {
        omp_set_num_threads(g_num_threads);
        openblas_set_num_threads(g_num_threads);
    }
    if (!g_pin_cores.empty()) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c : g_pin_cores) CPU_SET(c, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
}

static int g_pq_m = 8;      // number of PQ subquantizers
static int g_pq_nbits = 8;  // bits per subquantizer code

// --- binary index state ---
// IndexBinaryFlat / IndexBinaryIVF live in a completely separate FAISS class
// hierarchy from the float indices above (faiss::IndexBinary, not faiss::Index),
// so they need their own pointers. Only one of {g_index, g_bindex} is ever
// populated at a time; g_using_binary tracks which one is active.
static faiss::IndexBinary* g_bindex = nullptr;      // may point to IndexBinaryIDMap or base binary index
static faiss::IndexBinaryIDMap* g_binidmap = nullptr; // convenience pointer if g_bindex is an IndexBinaryIDMap
static bool g_using_binary = false;
static int g_bin_dim_bits = 0;  // vector dimensionality in BITS (must be a multiple of 8)
static int g_bin_nlist = 1;
static int g_bin_nprobe = 1;

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
    if (g_bindex) {
        try {
            delete g_bindex;
        } catch (...) {
            // ignore destructor exceptions
        }
        g_bindex = nullptr;
    }
    g_binidmap = nullptr;
    g_using_binary = false;
    g_bin_dim_bits = 0;
    g_bin_nlist = 1;
    g_bin_nprobe = 1;
    g_dim = 0;
    g_ivf_nprobe = 1;
    g_hnsw_M = 1;
    g_hnsw_ef_construction = 1;
    g_hnsw_ef_search = 1;
    g_pq_m = 8;
    g_pq_nbits = 8;
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

// Helper: compute next id from the binary id_map vector (if IndexBinaryIDMap present)
static void recompute_next_id_from_binidmap() {
    if (!g_binidmap) {
        LOGE("recompute_next_id_from_binidmap: IndexBinaryIDMap not present — cannot recompute next id");
        g_next_id = (faiss::idx_t)-1;
        return;
    }
    const std::vector<faiss::idx_t> &ids = g_binidmap->id_map;
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
    g_idmap->own_fields = true;
    g_index = g_idmap;
}

extern "C" {

static float* g_trainBuffer = nullptr;
static size_t g_trainBufferCount = 0; // number of floats, not vectors

/*
 * public native boolean allocTrainBufferNative(int numFloats);
 * Allocates a raw native float buffer of the given size. Does not touch
 * the JVM/ART heap — this is a pure C++ heap allocation.
 */
JNIEXPORT jboolean JNICALL
Java_com_example_faiss_FaissIndex_allocTrainBufferNative(
        JNIEnv* env, jobject thiz, jint numFloats) {

    if (g_trainBuffer != nullptr) {
        LOGE("allocTrainBufferNative: buffer already allocated");
        return JNI_FALSE;
    }

    g_trainBuffer = new (std::nothrow) float[numFloats];
    if (!g_trainBuffer) {
        LOGE("allocTrainBufferNative: failed to allocate %d floats (%zu bytes)",
             numFloats, (size_t)numFloats * sizeof(float));
        return JNI_FALSE;
    }

    g_trainBufferCount = (size_t)numFloats;
    LOGI("allocTrainBufferNative: allocated %d floats (%.1f MB) natively",
         numFloats, (double)(numFloats * sizeof(float)) / (1024.0 * 1024.0));
    return JNI_TRUE;
}
/*
 * public native void freeTrainBufferNative();
 * Frees the native training buffer. Safe to call when no buffer is allocated.
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_freeTrainBufferNative(JNIEnv* env, jobject thiz) {
    if (g_trainBuffer) {
        delete[] g_trainBuffer;
        g_trainBuffer = nullptr;
        g_trainBufferCount = 0;
        LOGI("freeTrainBufferNative: buffer freed");
    }
}
/*
 * public native boolean writeToBufferNative(float[] data, int offset, int count);
 * Copies `count` floats from the Java array into the native training buffer
 * starting at float-index `offset`. Called repeatedly with small batches
 * (see buildFaissIndex in ANNBenchmark.kt) to fill the buffer that
 * trainIvfFromBufferNative then trains on. Returns JNI_FALSE if the buffer
 * is not allocated or the range is out of bounds.
 */
JNIEXPORT jboolean JNICALL
Java_com_example_faiss_FaissIndex_writeToBufferNative(
        JNIEnv* env, jobject thiz, jfloatArray jdata, jint offset, jint count) {

    if (!g_trainBuffer) {
        LOGE("writeToBufferNative: no buffer allocated");
        return JNI_FALSE;
    }
    if ((size_t)offset + (size_t)count > g_trainBufferCount) {
        LOGE("writeToBufferNative: write would overflow buffer "
             "(offset=%d, count=%d, capacity=%zu)", offset, count, g_trainBufferCount);
        return JNI_FALSE;
    }

    jsize len = env->GetArrayLength(jdata);
    if ((size_t)len < (size_t)count) {
        LOGE("writeToBufferNative: source array too small (len=%d, needed=%d)", len, count);
        return JNI_FALSE;
    }

    jfloat* src = env->GetFloatArrayElements(jdata, nullptr);
    memcpy(g_trainBuffer + offset, src, (size_t)count * sizeof(float));
    env->ReleaseFloatArrayElements(jdata, src, JNI_ABORT); // read-only

    return JNI_TRUE;
}

/*
 * Diagnostic helper (not used by the benchmarks): copies floats from the native training buffer back into [out].
 */
JNIEXPORT jboolean JNICALL
Java_com_example_faiss_FaissIndex_readFromBufferNative(
        JNIEnv* env, jobject thiz, jfloatArray jout, jint offset, jint count) {

    if (!g_trainBuffer) {
        LOGE("readFromBufferNative: no buffer allocated");
        return JNI_FALSE;
    }
    if ((size_t)offset + (size_t)count > g_trainBufferCount) {
        LOGE("readFromBufferNative: read would overflow buffer");
        return JNI_FALSE;
    }

    jsize len = env->GetArrayLength(jout);
    if ((size_t)len < (size_t)count) {
        LOGE("readFromBufferNative: destination array too small");
        return JNI_FALSE;
    }

    env->SetFloatArrayRegion(jout, 0, count, g_trainBuffer + offset);
    return JNI_TRUE;
}
/*
 * public native void trainIvfFromBufferNative(int numVectors);
 * Trains the current IVF index using the native training buffer that was
 * previously allocated and filled via allocTrainBufferNative/writeToBufferNative.
 * Does not free the buffer - call freeTrainBufferNative separately once done.
 * Throws IllegalStateException if training cannot run or FAISS fails.
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_trainIvfFromBufferNative(
        JNIEnv* env, jobject thiz, jint numVectors) {

    // Log the problem and throw it to the Java caller, so a failed training is not
    // mistaken for a successful one.
    auto fail = [&](const std::string& msg) {
        LOGE("trainIvfFromBufferNative: %s", msg.c_str());
        jclass exc = env->FindClass("java/lang/IllegalStateException");
        if (exc) env->ThrowNew(exc, ("trainIvfFromBufferNative: " + msg).c_str());
    };

    if (!g_trainBuffer) {
        fail("no buffer allocated");
        return;
    }
    if (!g_index) {
        fail("index is null");
        return;
    }
    faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>( (g_idmap ? g_idmap->index : g_index) );
    if (!ivf) {
        fail("current index is not an IVF index");
        return;
    }
    if (g_dim <= 0) {
        fail("invalid dimension " + std::to_string(g_dim));
        return;
    }
    if (numVectors <= 0 || (size_t)numVectors * (size_t)g_dim > g_trainBufferCount) {
        fail("requested " + std::to_string(numVectors) + " vectors * dim " +
             std::to_string(g_dim) + " exceeds buffer capacity " +
             std::to_string(g_trainBufferCount) + " floats");
        return;
    }

    try {
        ivf->train((size_t)numVectors, g_trainBuffer);
        LOGI("trainIvfFromBufferNative: trained on %d vectors (dim=%d)", numVectors, g_dim);
    } catch (const std::exception &e) {
        fail(std::string("exception during ivf->train: ") + e.what());
    }
}

/** Diagnostic helper (not used by the benchmarks): returns the IVF centroids as a flat nlist * dim array.*/
JNIEXPORT jfloatArray JNICALL
Java_com_example_faiss_FaissIndex_getCentroidsNative(JNIEnv* env, jobject thiz) {
    faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>( (g_idmap ? g_idmap->index : g_index) );
    if (!ivf) {
        LOGE("getCentroidsNative: current index is not an IVF index");
        return nullptr;
    }

    faiss::IndexFlat* quantizer = dynamic_cast<faiss::IndexFlat*>(ivf->quantizer);
    if (!quantizer) {
        LOGE("getCentroidsNative: quantizer is not an IndexFlat");
        return nullptr;
    }

    size_t count = (size_t)quantizer->ntotal * (size_t)g_dim;
    jfloatArray result = env->NewFloatArray((jsize)count);
    if (!result) return nullptr;

    env->SetFloatArrayRegion(result, 0, (jsize)count,
                             reinterpret_cast<const jfloat*>(quantizer->get_xb()));

    return result;
}
/*
 * public native void setNumThreadsNative(int n);   n <= 0 -> library default
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_setNumThreadsNative(JNIEnv* env, jobject thiz, jint n) {
    save_thread_defaults();
    g_num_threads = n > 0 ? (int)n : 0;
    if (g_num_threads == 0) {
        omp_set_num_threads(g_default_omp_threads);
        openblas_set_num_threads(g_default_blas_threads);
    }
    apply_num_threads();
    LOGI("FAISS threads set to %d (0 = default)", g_num_threads);
}

/*
 * public native int setCpuAffinityNative(int[] cores);  empty array -> no pinning.
 * Returns 0 on success, otherwise errno from sched_setaffinity.
 */
JNIEXPORT jint JNICALL
Java_com_example_faiss_FaissIndex_setCpuAffinityNative(JNIEnv* env, jobject thiz, jintArray jcores) {
    save_thread_defaults();
    g_pin_cores.clear();
    jsize n = jcores ? env->GetArrayLength(jcores) : 0;
    if (n > 0) {
        std::vector<jint> tmp(n);
        env->GetIntArrayRegion(jcores, 0, n, tmp.data());
        for (jint c : tmp) if (c >= 0 && c < CPU_SETSIZE) g_pin_cores.push_back((int)c);
    }
    if (g_pin_cores.empty()) {
        sched_setaffinity(0, sizeof(g_default_affinity), &g_default_affinity);
        return 0;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : g_pin_cores) CPU_SET(c, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        int err = errno;
        LOGE("sched_setaffinity failed: %s", std::strerror(err));
        g_pin_cores.clear();
        return (jint)err;
    }
    return 0;
}

/*
 * public native int getCurrentCpuNative();  core the calling thread runs on right now
 */
JNIEXPORT jint JNICALL
Java_com_example_faiss_FaissIndex_getCurrentCpuNative(JNIEnv* env, jobject thiz) {
    apply_num_threads();
    return (jint)sched_getcpu();
}

/*
 * public native int getNumThreadsNative();  threads OpenMP will use on the calling thread
 */
JNIEXPORT jint JNICALL
Java_com_example_faiss_FaissIndex_getNumThreadsNative(JNIEnv* env, jobject thiz) {
    apply_num_threads();
    return (jint)omp_get_max_threads();
}

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
    g_idmap->own_fields = true;
    g_index = g_idmap;
    g_next_id = 0;
    // LOGI("Initialized %s with dim=%d", indexType.c_str(), dim);
}
/*
 * public native void initIvfNative(int dim, int nlist, int nprobe, int metric);
 * Creates an empty IVF-Flat index (full-precision vectors) wrapped in IndexIDMap.
 * The index must still be trained before vectors are added.
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
    ivf->own_fields = true;  // IVF frees its quantizer

    // Wrap in IDMap to control ids and preserve them when serializing
    g_idmap = new faiss::IndexIDMap(ivf);
    g_idmap->own_fields = true;  // IDMap frees the IVF index
    g_index = g_idmap;

    g_next_id = 0;
    // LOGI("Initialized IndexIVFFlat dim=%d nlist=%d nprobe=%d metric=%d", dim, nlist, nprobe, metric);
}

/*
 * public native void initIvfSq8Native(int dim, int nlist, int nprobe, int metric);
 *
 * IndexIVFScalarQuantizer (QT_8bit): same cluster-then-search structure as IVF-Flat, but
 * each vector is stored as 1 byte per dimension instead of 4 (about 4x smaller index, small
 * recall loss). Training still uses full-precision vectors; vectors are compressed when
 * they are added. The index must be trained before vectors are added.
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_initIvfSq8Native(JNIEnv* env, jobject thiz, jint dim,
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

    try {
        faiss::IndexIVFScalarQuantizer* ivfsq = new faiss::IndexIVFScalarQuantizer(
                quantizer, (size_t)dim, (size_t)nlist, faiss::ScalarQuantizer::QT_8bit, mt);
        ivfsq->nprobe = g_ivf_nprobe;

        // Wrap in IDMap to control ids and preserve them when serializing
        ivfsq->own_fields = true;  // IVF frees its quantizer
        g_idmap = new faiss::IndexIDMap(ivfsq);
        g_idmap->own_fields = true;  // IDMap frees the IVF index
        g_index = g_idmap;
        g_next_id = 0;
        LOGI("initIvfSq8Native: dim=%d nlist=%d nprobe=%d", (int)dim, (int)nlist, (int)nprobe);
    } catch (const std::exception &e) {
        LOGE("Exception creating IVFSQ8 index: %s", e.what());
        delete quantizer;
    }
}
/*
 * public native void initIvfPqNative(int dim, int nlist, int nprobe, int m, int nbits, int metric);
 *
 * IndexIVFPQ: same cluster-then-search structure as IVF-Flat, but each residual vector is
 * compressed with Product Quantization (m subquantizers, nbits bits per subquantizer code)
 * instead of being stored in full precision. Requires dim % m == 0.
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_initIvfPqNative(JNIEnv* env, jobject thiz, jint dim,
                                                   jint nlist, jint nprobe, jint m, jint nbits,
                                                   jint metric) {
    delete_index();
    g_dim = dim;
    g_ivf_nprobe = (int)nprobe;
    g_pq_m = (int)m;
    g_pq_nbits = (int)nbits;

    if (m <= 0 || dim % m != 0) {
        LOGE("initIvfPqNative: dim (%d) must be divisible by m (%d)", dim, m);
        return;
    }

    faiss::MetricType mt = faiss::METRIC_L2;
    if (metric == 1) mt = faiss::METRIC_INNER_PRODUCT;

    faiss::IndexFlat* quantizer = nullptr;
    if (mt == faiss::METRIC_L2) {
        quantizer = new faiss::IndexFlatL2(dim);
    } else {
        quantizer = new faiss::IndexFlatIP(dim);
    }

    try {
        faiss::IndexIVFPQ* ivfpq = new faiss::IndexIVFPQ(
                quantizer, (size_t)dim, (size_t)nlist, (size_t)m, (size_t)nbits, mt);
        ivfpq->nprobe = g_ivf_nprobe;

        // Wrap in IDMap to control ids and preserve them when serializing
        ivfpq->own_fields = true;
        g_idmap = new faiss::IndexIDMap(ivfpq);
        g_idmap->own_fields = true;
        g_index = g_idmap;
        g_next_id = 0;
        // LOGI("Initialized IndexIVFPQ dim=%d nlist=%d nprobe=%d m=%d nbits=%d metric=%d",
        //      dim, nlist, nprobe, m, nbits, metric);
    } catch (const std::exception &e) {
        LOGE("Exception creating IVFPQ index: %s", e.what());
        delete quantizer;
    }
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
        g_idmap->own_fields = true;
        g_index = g_idmap;
        g_next_id = 0;
        /* LOGI("Initialized IndexHNSWFlat dim=%d M=%d efConstruction=%d efSearch=%d metric=%d",
             dim, M, efConstruction, efSearch, metric); */
    } catch (const std::exception &e) {
        LOGE("Exception creating HNSW index: %s", e.what());
    }
}

/*
 * public native void initBinaryFlatNative(int dimBits);
 *
 * IndexBinaryFlat: exhaustive Hamming-distance search over packed-bit codes.
 * dimBits is the vector dimensionality in BITS (must be a multiple of 8) — each
 * embedding float is sign-thresholded into one bit on the Kotlin side before
 * being passed down here as packed bytes.
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_initBinaryFlatNative(JNIEnv* env, jobject thiz, jint dimBits) {
    delete_index();

    if (dimBits <= 0 || dimBits % 8 != 0) {
        LOGE("initBinaryFlatNative: dimBits (%d) must be a positive multiple of 8", dimBits);
        return;
    }

    try {
        faiss::IndexBinaryFlat* flat = new faiss::IndexBinaryFlat((faiss::idx_t)dimBits);
        g_binidmap = new faiss::IndexBinaryIDMap(flat);
        g_binidmap->own_fields = true;
        g_bindex = g_binidmap;
        g_bin_dim_bits = dimBits;
        g_using_binary = true;
        g_next_id = 0;
        // LOGI("Initialized IndexBinaryFlat dimBits=%d", dimBits);
    } catch (const std::exception &e) {
        LOGE("Exception creating IndexBinaryFlat: %s", e.what());
    }
}

/*
 * public native void initBinaryIvfNative(int dimBits, int nlist, int nprobe);
 *
 * IndexBinaryIVF: same cluster-then-search structure as IVF-Flat, but the coarse
 * quantizer and stored codes both operate on packed bits with Hamming distance
 * instead of full-precision vectors with L2/IP distance.
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_initBinaryIvfNative(JNIEnv* env, jobject thiz, jint dimBits,
                                                       jint nlist, jint nprobe) {
    delete_index();

    if (dimBits <= 0 || dimBits % 8 != 0) {
        LOGE("initBinaryIvfNative: dimBits (%d) must be a positive multiple of 8", dimBits);
        return;
    }

    g_bin_dim_bits = dimBits;
    g_bin_nlist = (int)nlist;
    g_bin_nprobe = (int)nprobe;

    faiss::IndexBinaryFlat* quantizer = new faiss::IndexBinaryFlat((faiss::idx_t)dimBits);

    try {
        faiss::IndexBinaryIVF* ivf = new faiss::IndexBinaryIVF(quantizer, (size_t)dimBits, (size_t)nlist);
        ivf->nprobe = (size_t)nprobe;
        ivf->own_fields = true;  // binary IVF frees its quantizer

        g_binidmap = new faiss::IndexBinaryIDMap(ivf);
        g_binidmap->own_fields = true;
        g_bindex = g_binidmap;
        g_using_binary = true;
        g_next_id = 0;
        // LOGI("Initialized IndexBinaryIVF dimBits=%d nlist=%d nprobe=%d", dimBits, nlist, nprobe);
    } catch (const std::exception &e) {
        LOGE("Exception creating IndexBinaryIVF: %s", e.what());
        delete quantizer;
    }
}

/*
 * public native void trainBinaryIvfNative(byte[] packedTrainingVectors, int numVectors);
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_trainBinaryIvfNative(JNIEnv* env, jobject thiz, jbyteArray jtrain, jint numVectors) {
    if (!g_bindex) {
        LOGE("trainBinaryIvfNative called but binary index is null");
        return;
    }
    faiss::IndexBinaryIVF* ivf = dynamic_cast<faiss::IndexBinaryIVF*>(g_binidmap ? g_binidmap->index : g_bindex);
    if (!ivf) {
        LOGE("trainBinaryIvfNative called but current index is not a Binary IVF index");
        return;
    }
    if (g_bin_dim_bits <= 0) {
        LOGE("Invalid binary dim (%d) for training", g_bin_dim_bits);
        return;
    }

    const int codeSize = g_bin_dim_bits / 8;
    jsize len = env->GetArrayLength(jtrain);
    jbyte* data = env->GetByteArrayElements(jtrain, nullptr);

    if (len < numVectors * codeSize) {
        LOGE("trainBinaryIvfNative: provided array length %d is less than numVectors*codeSize (%d)", len, numVectors * codeSize);
        env->ReleaseByteArrayElements(jtrain, data, 0);
        return;
    }

    apply_num_threads();
    try {
        ivf->train((size_t)numVectors, (const uint8_t*)data);
        // LOGI("Binary IVF trained with %d vectors (dimBits=%d)", numVectors, g_bin_dim_bits);
    } catch (const std::exception &e) {
        LOGE("Exception during binary ivf->train: %s", e.what());
    }

    env->ReleaseByteArrayElements(jtrain, data, 0);
}

/*
 * public native long[] addBinaryNative(byte[] packedVecs, int batchSize);
 */
JNIEXPORT jlongArray JNICALL
Java_com_example_faiss_FaissIndex_addBinaryNative(JNIEnv* env, jobject thiz, jbyteArray jvecs, jint batchSize) {
    if (!g_binidmap || g_bin_dim_bits <= 0) {
        return nullptr;
    }

    const int codeSize = g_bin_dim_bits / 8;
    const jsize totalLen = env->GetArrayLength(jvecs);
    if (totalLen < batchSize * codeSize) {
        LOGE("addBinaryNative: array too small");
        return nullptr;
    }

    jbyte* data = env->GetByteArrayElements(jvecs, nullptr);

    std::vector<faiss::idx_t> ids(batchSize);
    for (int i = 0; i < batchSize; i++) {
        ids[i] = g_next_id++;
    }

    apply_num_threads();
    try {
        g_binidmap->add_with_ids(
                batchSize,
                (const uint8_t*)data,
                ids.data()
        );
    } catch (const std::exception &e) {
        LOGE("Exception in addBinaryNative: %s", e.what());
        env->ReleaseByteArrayElements(jvecs, data, 0);
        return nullptr;
    }

    env->ReleaseByteArrayElements(jvecs, data, 0);

    jlongArray result = env->NewLongArray(batchSize);
    std::vector<jlong> out(batchSize);
    for (int i = 0; i < batchSize; i++) {
        out[i] = (jlong)ids[i];
    }
    env->SetLongArrayRegion(result, 0, batchSize, out.data());

    return result;
}

/*
 * public native FaissIndex.QueryResults queryBinaryNative(byte[] packedQueries, int queryCount, int k);
 *
 * Hamming distances are native int32; they're widened to float here so the result
 * can reuse the same QueryResults(long[], float[]) JNI constructor as the float
 * query path — no change needed on the Kotlin QueryResults type.
 */
JNIEXPORT jobject JNICALL
Java_com_example_faiss_FaissIndex_queryBinaryNative(JNIEnv* env, jobject thiz, jbyteArray jqueries, jint queryCount, jint k) {
    if (!g_bindex) {
        LOGE("queryBinaryNative called but binary index is null");
        return nullptr;
    }

    const int codeSize = g_bin_dim_bits / 8;
    const jsize totalLen = env->GetArrayLength(jqueries);
    if (totalLen < queryCount * codeSize) {
        LOGE("queryBinaryNative: array too small");
        return nullptr;
    }

    jbyte* qptr = env->GetByteArrayElements(jqueries, nullptr);

    std::vector<int32_t> distances(queryCount * k);
    std::vector<faiss::idx_t> labels(queryCount * k);

    faiss::IndexBinary* baseIndex = g_binidmap ? g_binidmap->index : g_bindex;

    faiss::IndexBinaryIVF* ivf = dynamic_cast<faiss::IndexBinaryIVF*>(baseIndex);
    if (ivf) {
        ivf->nprobe = (size_t)g_bin_nprobe;
    }

    apply_num_threads();
    try {
        baseIndex->search(
                queryCount,
                (const uint8_t*)qptr,
                k,
                distances.data(),
                labels.data()
        );
    } catch (const std::exception &e) {
        LOGE("Exception during binary search: %s", e.what());
        env->ReleaseByteArrayElements(jqueries, qptr, 0);
        return nullptr;
    }

    env->ReleaseByteArrayElements(jqueries, qptr, 0);

    jlongArray jlabels = env->NewLongArray(queryCount * k);
    std::vector<jlong> labels_j(queryCount * k);
    for (int i = 0; i < queryCount * k; ++i) {
        labels_j[i] = (jlong)labels[i];
    }
    env->SetLongArrayRegion(jlabels, 0, labels_j.size(), labels_j.data());

    jfloatArray jdistances = env->NewFloatArray(queryCount * k);
    std::vector<jfloat> distances_f(queryCount * k);
    for (int i = 0; i < queryCount * k; ++i) {
        distances_f[i] = (jfloat)distances[i];
    }
    env->SetFloatArrayRegion(jdistances, 0, distances_f.size(), distances_f.data());

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

    apply_num_threads();
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

    apply_num_threads();
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

    apply_num_threads();
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
    g_bin_nprobe = (int)nprobe;   // binary IVF reads this before every search
    faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>( (g_idmap ? g_idmap->index : g_index) );
    if (ivf) {
        ivf->nprobe = g_ivf_nprobe;
        // LOGI("setNprobe: updated ivf->nprobe=%d", g_ivf_nprobe);
    } else {
        LOGE("setNprobe: index is not IVF; stored nprobe=%d will be used if IVF created", g_ivf_nprobe);
    }
}

/*
 * public native void setEfSearchNative(int efSearch);  HNSW search-time setting
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_setEfSearchNative(JNIEnv* env, jobject thiz, jint efSearch) {
    g_hnsw_ef_search = (int)efSearch;
    faiss::Index* base = g_idmap ? g_idmap->index : g_index;
    if (auto hnsw = dynamic_cast<faiss::IndexHNSWFlat*>(base)) {
        hnsw->hnsw.efSearch = g_hnsw_ef_search;
    }
}

/*
 * public native long getTotalNative();
 */
JNIEXPORT jlong JNICALL
Java_com_example_faiss_FaissIndex_getTotalNative(JNIEnv* env, jobject thiz) {
    if (g_using_binary) {
        return g_bindex ? (jlong)g_bindex->ntotal : (jlong)0;
    }
    if (!g_index) return (jlong)0;
    return (jlong)g_index->ntotal;
}

/*
 * public native int getIndexTypeNative();
 * Returns the type of the current index, read from the index object itself:
 *   0 = none/unknown, 1 = flat, 2 = IVF-Flat, 3 = HNSW, 4 = IVFPQ, 5 = IVF-SQ8,
 *   6 = binary flat, 7 = binary IVF.
 * Binary indexes live in g_bindex, so they are checked first.
 * These numbers must match the `when` in FaissIndex.loadFrom().
 */
JNIEXPORT jint JNICALL
Java_com_example_faiss_FaissIndex_getIndexTypeNative(JNIEnv* env, jobject thiz) {
    if (g_bindex) {
        faiss::IndexBinary* bcore = g_bindex;
        if (auto* bmap = dynamic_cast<faiss::IndexBinaryIDMap*>(g_bindex)) bcore = bmap->index;
        if (dynamic_cast<faiss::IndexBinaryIVF*>(bcore)) return 7;
        if (dynamic_cast<faiss::IndexBinaryFlat*>(bcore)) return 6;
        return 0;
    }
    if (!g_index) return 0;
    faiss::Index* core = g_idmap ? g_idmap->index : g_index;
    // IVFPQ and SQ8 are subclasses of IndexIVF, so they must be tested before it
    if (dynamic_cast<faiss::IndexIVFPQ*>(core)) return 4;
    if (dynamic_cast<faiss::IndexIVFScalarQuantizer*>(core)) return 5;
    if (dynamic_cast<faiss::IndexIVF*>(core)) return 2;
    if (dynamic_cast<faiss::IndexHNSWFlat*>(core)) return 3;
    if (dynamic_cast<faiss::IndexFlat*>(core)) return 1;
    return 0;
}


/*
 * Persist FAISS index to path (string)
 * Java signature: public native void writeIndexNative(String path);
 */
JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_writeIndexNative(JNIEnv* env, jobject thiz, jstring jpath) {

    if (!g_index && !g_bindex) {
        LOGE("writeIndexNative: no index to write");
        // Throw to make Java handle fatal error
        jclass exc = env->FindClass("java/lang/IllegalStateException");
        env->ThrowNew(exc, "writeIndexNative: no index to write");
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

    if (g_using_binary) {
        // Require that the current binary index is an IndexBinaryIDMap.
        if (!g_binidmap) {
            LOGE("writeIndexNative: current binary index is not an IndexBinaryIDMap; refusing to write in strict-id mode");
            env->ReleaseStringUTFChars(jpath, path);
            jclass exc = env->FindClass("java/lang/IllegalStateException");
            env->ThrowNew(exc, "writeIndexNative: binary index does not preserve IDs (not IndexBinaryIDMap)");
            return;
        }

        try {
            faiss::write_index_binary(g_bindex, tmp.c_str());
            if (std::rename(tmp.c_str(), pathStr.c_str()) != 0) {
                std::remove(tmp.c_str());
                env->ReleaseStringUTFChars(jpath, path);
                std::string msg = std::string("writeIndexNative: failed to rename temp file to final location: ") + std::strerror(errno);
                jclass exc = env->FindClass("java/io/IOException");
                env->ThrowNew(exc, msg.c_str());
                return;
            }

            std::ofstream of(metaf.c_str(), std::ios::out | std::ios::trunc);
            if (!of.is_open()) {
                std::remove(pathStr.c_str());
                env->ReleaseStringUTFChars(jpath, path);
                std::string msg = std::string("writeIndexNative: could not open meta file for writing: ") + metaf;
                jclass exc = env->FindClass("java/io/IOException");
                env->ThrowNew(exc, msg.c_str());
                return;
            }

            of << "index_kind=binary\n";
            of << "g_bin_dim_bits=" << g_bin_dim_bits << "\n";
            of << "g_bin_nlist=" << g_bin_nlist << "\n";
            of << "g_bin_nprobe=" << g_bin_nprobe << "\n";
        } catch (const std::exception &e) {
            LOGE("Exception writing binary index: %s", e.what());
            env->ReleaseStringUTFChars(jpath, path);
            jclass exc = env->FindClass("java/io/IOException");
            env->ThrowNew(exc, std::string("writeIndexNative: error writing temp binary index: " + std::string(e.what())).c_str());
            return;
        }

        env->ReleaseStringUTFChars(jpath, path);
        return;
    }

    // Require that the current index is an IndexIDMap.
    faiss::IndexIDMap* idmap = dynamic_cast<faiss::IndexIDMap*>(g_index);
    if (!idmap) {
        LOGE("writeIndexNative: current index is not an IndexIDMap; refusing to write in strict-id mode");
        env->ReleaseStringUTFChars(jpath, path);
        jclass exc = env->FindClass("java/lang/IllegalStateException");
        env->ThrowNew(exc, "writeIndexNative: index does not preserve IDs (not IndexIDMap)");
        return;
    }

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
        of << "index_kind=float\n";
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

    // --- Read the meta file FIRST so we know which FAISS reader (float vs binary)
    //     to hand the index file to. "index_kind" is absent in meta files written
    //     before binary-index support existed; treat that as "float" for backward
    //     compatibility with indices saved by earlier builds. ---
    std::ifstream inf(metaf.c_str());
    if (!inf.is_open()) {
        env->ReleaseStringUTFChars(jpath, path);
        std::string msg = std::string("readIndexNative: failed to open meta file: ") + metaf;
        jclass exc = env->FindClass("java/io/IOException");
        env->ThrowNew(exc, msg.c_str());
        return JNI_FALSE;
    }

    std::string indexKind = "float";
    int new_dim = 0, new_ivf_nprobe = 0, new_hnsw_M = 0, new_hnsw_ef_construction = 0, new_hnsw_ef_search = 0;
    int new_bin_dim_bits = 0, new_bin_nlist = 0, new_bin_nprobe = 0;
    bool has_ivf_nprobe = false, has_hnsw_M = false, has_hnsw_ef_construction = false, has_hnsw_ef_search = false;
    bool has_bin_dim_bits = false, has_bin_nlist = false, has_bin_nprobe = false;

    std::string line;
    while (std::getline(inf, line)) {
        if (line.empty()) continue;
        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string key = line.substr(0, eq);
        std::string val = line.substr(eq + 1);
        try {
            if (key == "index_kind") {
                indexKind = val;
            } else if (key == "g_dim") {
                new_dim = std::stoi(val);
            } else if (key == "g_ivf_nprobe") {
                new_ivf_nprobe = std::stoi(val);
                has_ivf_nprobe = true;
            } else if (key == "g_hnsw_M") {
                new_hnsw_M = std::stoi(val);
                has_hnsw_M = true;
            } else if (key == "g_hnsw_ef_construction") {
                new_hnsw_ef_construction = std::stoi(val);
                has_hnsw_ef_construction = true;
            } else if (key == "g_hnsw_ef_search") {
                new_hnsw_ef_search = std::stoi(val);
                has_hnsw_ef_search = true;
            } else if (key == "g_bin_dim_bits") {
                new_bin_dim_bits = std::stoi(val);
                has_bin_dim_bits = true;
            } else if (key == "g_bin_nlist") {
                new_bin_nlist = std::stoi(val);
                has_bin_nlist = true;
            } else if (key == "g_bin_nprobe") {
                new_bin_nprobe = std::stoi(val);
                has_bin_nprobe = true;
            }
        } catch (...) {
            inf.close();
            env->ReleaseStringUTFChars(jpath, path);
            jclass exc = env->FindClass("java/io/IOException");
            env->ThrowNew(exc, std::string("readIndexNative: parse error in meta file for key " + key).c_str());
            return JNI_FALSE;
        }
    }
    inf.close();

    if (indexKind == "binary") {
        if (!has_bin_dim_bits || !has_bin_nlist || !has_bin_nprobe) {
            env->ReleaseStringUTFChars(jpath, path);
            jclass exc = env->FindClass("java/io/IOException");
            env->ThrowNew(exc, "readIndexNative: binary meta file missing one or more required parameters (g_bin_dim_bits, g_bin_nlist, g_bin_nprobe)");
            return JNI_FALSE;
        }

        faiss::IndexBinary* new_bindex = nullptr;
        try {
            new_bindex = faiss::read_index_binary(pathStr.c_str());
        } catch (const std::exception &e) {
            LOGE("Exception reading binary index %s: %s", path, e.what());
            env->ReleaseStringUTFChars(jpath, path);
            jclass exc = env->FindClass("java/io/IOException");
            env->ThrowNew(exc, std::string("readIndexNative: error reading binary index: " + std::string(e.what())).c_str());
            return JNI_FALSE;
        }

        faiss::IndexBinaryIDMap* loaded_binidmap = dynamic_cast<faiss::IndexBinaryIDMap*>(new_bindex);
        if (!loaded_binidmap) {
            LOGE("readIndexNative: loaded binary index is not an IndexBinaryIDMap; refusing to load in strict-id mode");
            try { delete new_bindex; } catch (...) {}
            env->ReleaseStringUTFChars(jpath, path);
            jclass exc = env->FindClass("java/lang/IllegalStateException");
            env->ThrowNew(exc, "readIndexNative: loaded binary index does not preserve IDs (not IndexBinaryIDMap)");
            return JNI_FALSE;
        }

        delete_index(); // safely tears down whatever was previously active (float or binary)
        g_bindex = loaded_binidmap; // ownership transferred
        g_binidmap = loaded_binidmap;
        g_using_binary = true;
        g_bin_dim_bits = new_bin_dim_bits;
        g_bin_nlist = new_bin_nlist;
        g_bin_nprobe = new_bin_nprobe;

        if (auto bivf = dynamic_cast<faiss::IndexBinaryIVF*>(g_binidmap->index)) {
            bivf->nprobe = (size_t)g_bin_nprobe;
        }

        recompute_next_id_from_binidmap();
        if (g_next_id == (faiss::idx_t)-1) {
            LOGE("readIndexNative: failed to recompute next id from IndexBinaryIDMap");
            env->ReleaseStringUTFChars(jpath, path);
            jclass exc = env->FindClass("java/lang/IllegalStateException");
            env->ThrowNew(exc, "readIndexNative: failed to recompute next id");
            return JNI_FALSE;
        }

        env->ReleaseStringUTFChars(jpath, path);
        return JNI_TRUE;
    }

    // ---- float-index read path ----
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

    // For strictness, require all four legacy keys always. Checked BEFORE the current
    // index is replaced: new_index is still owned only by this function here, so deleting
    // it cannot leave g_index / g_idmap dangling (and clear() cannot free it a second time).
    if (!has_ivf_nprobe || !has_hnsw_M || !has_hnsw_ef_construction || !has_hnsw_ef_search) {
        try { delete new_index; } catch (...) {}
        env->ReleaseStringUTFChars(jpath, path);
        jclass exc = env->FindClass("java/io/IOException");
        env->ThrowNew(exc, "readIndexNative: meta file missing one or more required parameters (g_ivf_nprobe, g_hnsw_M, g_hnsw_ef_construction, g_hnsw_ef_search)");
        return JNI_FALSE;
    }

    // Replace current index with the loaded IndexIDMap
    delete_index(); // deletes existing g_index safely
    g_index = loaded_idmap; // ownership transferred
    g_idmap = loaded_idmap;

    g_dim = new_dim;
    g_ivf_nprobe = new_ivf_nprobe;
    g_hnsw_M = new_hnsw_M;
    g_hnsw_ef_construction = new_hnsw_ef_construction;
    g_hnsw_ef_search = new_hnsw_ef_search;

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
    if (g_using_binary) {
        if (!g_bindex) {
            return env->NewStringUTF("no index loaded");
        }
        std::ostringstream boss;
        boss << "ntotal=" << g_bindex->ntotal;
        faiss::IndexBinary* bcore = g_binidmap ? g_binidmap->index : g_bindex;
        if (auto bivf = dynamic_cast<faiss::IndexBinaryIVF*>(bcore)) {
            boss << " type=BinaryIVF";
            boss << " d=" << bivf->d;
            boss << " nlist=" << bivf->nlist;
            boss << " nprobe=" << bivf->nprobe;
            long nonempty = 0;
            for (size_t i = 0; i < bivf->nlist; ++i) {
                if (bivf->invlists->list_size(i) > 0) ++nonempty;
            }
            boss << " nonempty_lists=" << nonempty;
        } else if (auto bflat = dynamic_cast<faiss::IndexBinaryFlat*>(bcore)) {
            boss << " type=BinaryFlat d=" << bflat->d;
        } else {
            boss << " type=unknown-binary";
        }
        std::string bs = boss.str();
        return env->NewStringUTF(bs.c_str());
    }

    if (!g_index) {
        return env->NewStringUTF("no index loaded");
    }

    std::ostringstream oss;
    oss << "ntotal=" << g_index->ntotal;

    // try IndexIDMap wrapper
    faiss::Index* core = nullptr;
    if (g_idmap) core = g_idmap->index; else core = g_index;

    if (auto ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(core)) {
        oss << " type=IVFPQ";
        oss << " d=" << ivfpq->d;
        oss << " nlist=" << ivfpq->nlist;
        oss << " nprobe=" << ivfpq->nprobe;
        oss << " m=" << ivfpq->pq.M;
        oss << " nbits=" << ivfpq->pq.nbits;
        long nonempty = 0;
        for (int i = 0; i < ivfpq->nlist; ++i) {
            if (ivfpq->invlists->list_size(i) > 0) ++nonempty;
        }
        oss << " nonempty_lists=" << nonempty;
    } else if (auto ivf = dynamic_cast<faiss::IndexIVF*>(core)) {
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
            oss << "\"ivf_lists_scanned\":" << faiss::indexIVF_stats.nlist << ",";
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
    oss << "\"ntotal\":" << g_index->ntotal;
    oss << "}";
    return env->NewStringUTF(oss.str().c_str());
}

JNIEXPORT jstring JNICALL
Java_com_example_faiss_FaissIndex_getIvfPqStatsNative(JNIEnv* env, jobject thiz) {
    std::string json = "{}";
    try {
        faiss::Index* core = (g_idmap ? g_idmap->index : g_index);
        faiss::IndexIVFPQ* ivfpq = dynamic_cast<faiss::IndexIVFPQ*>(core);
        std::ostringstream oss;
        if (ivfpq) {
            long nonempty = 0;
            for (int i = 0; i < ivfpq->nlist; ++i) {
                if (ivfpq->invlists->list_size(i) > 0) ++nonempty;
            }
            oss << "{";
            oss << "\"ntotal\":" << g_index->ntotal << ",";
            oss << "\"nlist\":" << ivfpq->nlist << ",";
            oss << "\"nprobe\":" << ivfpq->nprobe << ",";
            oss << "\"m\":" << ivfpq->pq.M << ",";
            oss << "\"nbits\":" << ivfpq->pq.nbits << ",";
            oss << "\"nonempty_lists\":" << nonempty << ",";
#ifdef FAISS_HAS_INDEX_IVF_STATS
            oss << "\"ivf_nq\":" << faiss::indexIVF_stats.nq << ",";
            oss << "\"ivf_ndis\":" << faiss::indexIVF_stats.ndis << ",";
            oss << "\"ivf_search_time_ms\":" << faiss::indexIVF_stats.search_time << ",";
            oss << "\"ivf_quant_time_ms\":" << faiss::indexIVF_stats.quantization_time;
#else
            oss << "\"note\":\"ivf-stats-not-available\"";
#endif
            oss << "}";
            json = oss.str();
        } else {
            oss << "{\"type\":\"not-ivfpq\",\"ntotal\":" << (g_index ? g_index->ntotal : 0) << "}";
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
Java_com_example_faiss_FaissIndex_getBinaryStatsNative(JNIEnv* env, jobject thiz) {
    std::string json = "{}";
    try {
        std::ostringstream oss;
        if (!g_bindex) {
            oss << "{\"error\":\"no-index\"}";
            return env->NewStringUTF(oss.str().c_str());
        }
        faiss::IndexBinary* bcore = g_binidmap ? g_binidmap->index : g_bindex;
        oss << "{";
        oss << "\"ntotal\":" << g_bindex->ntotal << ",";
        if (auto bivf = dynamic_cast<faiss::IndexBinaryIVF*>(bcore)) {
            long nonempty = 0;
            for (size_t i = 0; i < bivf->nlist; ++i) {
                if (bivf->invlists->list_size(i) > 0) ++nonempty;
            }
            oss << "\"type\":\"BinaryIVF\",";
            oss << "\"nlist\":" << bivf->nlist << ",";
            oss << "\"nprobe\":" << bivf->nprobe << ",";
            oss << "\"nonempty_lists\":" << nonempty;
        } else if (dynamic_cast<faiss::IndexBinaryFlat*>(bcore)) {
            oss << "\"type\":\"BinaryFlat\"";
        } else {
            oss << "\"type\":\"unknown-binary\"";
        }
        oss << "}";
        json = oss.str();
    } catch (const std::exception& e) {
        std::ostringstream oss;
        oss << "{\"error\":\"" << e.what() << "\"}";
        json = oss.str();
    }
    return env->NewStringUTF(json.c_str());
}

JNIEXPORT void JNICALL
Java_com_example_faiss_FaissIndex_clearNative(JNIEnv* env, jobject thiz) {
    delete_index();
    LOGI("Cleared FAISS index");
}

/*
 * public native boolean isBinaryActiveNative();
 *
 * Lets the Kotlin layer discover whether the currently active index is a binary
 * index after a loadFrom() call, since readIndexNative() (unlike the initXxx()
 * calls) has no matching Kotlin-side call site to set currentIndexType directly.
 */
/*
 * public native int getBinaryDimBitsNative();  dimension (in bits = floats per vector) of the
 * active binary index, or 0 when none is active. Covers indexes created by init and by loadFrom.
 */
JNIEXPORT jint JNICALL
Java_com_example_faiss_FaissIndex_getBinaryDimBitsNative(JNIEnv* env, jobject thiz) {
    return (g_using_binary && g_bindex) ? (jint)g_bin_dim_bits : (jint)0;
}

JNIEXPORT jboolean JNICALL
Java_com_example_faiss_FaissIndex_isBinaryActiveNative(JNIEnv* env, jobject thiz) {
    return g_using_binary ? JNI_TRUE : JNI_FALSE;
}

} // extern "C"
