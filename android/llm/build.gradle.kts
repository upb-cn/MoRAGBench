plugins {
    id("com.android.library")
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.example.local_llm"
    compileSdk = 35

    defaultConfig {
        minSdk = 24

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

        ndk {
            abiFilters += listOf("arm64-v8a") // add "armeabi-v7a" if you included it
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    kotlinOptions {
        jvmTarget = "11"
    }
    buildFeatures {
        viewBinding = false
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)

    // AndroidX Lifecycle
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.1")

    // Compose Activity integration
    implementation("androidx.activity:activity-compose:1.10.1")

    // Compose BOM (Bill of Materials)
    implementation(platform("androidx.compose:compose-bom:2025.06.00"))

    // Core Compose UI libraries
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")

    // Material Design 3
    implementation("androidx.compose.material3:material3")

    // ONNX Runtime (Android)
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.20.0")

    // org.json library
    implementation("org.json:json:20240303")

    // Compose testing (Android Instrumentation Tests)
    androidTestImplementation(platform("androidx.compose:compose-bom:2025.06.00"))
    androidTestImplementation("androidx.compose.ui:ui-test-junit4")

    // Debug-only tools for Compose UI inspection
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation(libs.androidx.appcompat)
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit)
    androidTestImplementation(libs.androidx.espresso.core)
    implementation("androidx.constraintlayout:constraintlayout:2.2.1")
    implementation(files("libs/onnxruntime-genai-android-0.7.1.aar"))
    implementation("io.noties.markwon:core:4.6.2")
    implementation("androidx.recyclerview:recyclerview:1.4.0")
    implementation ("com.google.android.material:material:1.12.0")
}