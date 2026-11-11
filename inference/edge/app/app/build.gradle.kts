plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.compose)
}

android {
    namespace = "com.example.edgeai"
    compileSdk {
        version = release(37)
    }

    defaultConfig {
        applicationId = "com.example.edgeai"
        minSdk = 24
        targetSdk = 37
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            optimization {
                enable = true
                packageScope = setOf("androidx.**", "kotlin.**", "kotlinx.**")
            }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }
    buildFeatures {
        compose = true
    }
}

dependencies {
    val camerax_version = "1.6.2"
    // The following line is optional, as the core library is included indirectly by camera-camera2
    implementation ("androidx.camera:camera-core:${camerax_version}")
    implementation ("androidx.camera:camera-camera2:${camerax_version}")
    // If you want to additionally use the CameraX Lifecycle library
    implementation ("androidx.camera:camera-lifecycle:${camerax_version}")
    // If you want to additionally use the CameraX VideoCapture library
    implementation ("androidx.camera:camera-video:${camerax_version}")
    // If you want to additionally use the CameraX View class
    implementation ("androidx.camera:camera-view:${camerax_version}")
    // If you want to additionally add CameraX ML Kit Vision Integration
    implementation ("androidx.camera:camera-mlkit-vision:${camerax_version}")
    // If you want to additionally use the CameraX Extensions library

    implementation("com.google.mediapipe:tasks-vision:latest.release")

    implementation("com.google.ai.edge.litert:litert:2.3.0")
// Optional: include for GPU acceleration
    implementation("com.google.ai.edge.litert:litert-gpu:2.3.0")


    implementation("io.coil-kt.coil3:coil-compose:3.6.3")

    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.activity.compose)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.ui.graphics)
    implementation(libs.androidx.compose.ui.tooling.preview)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    testImplementation(libs.junit)
    androidTestImplementation(platform(libs.androidx.compose.bom))
    androidTestImplementation(libs.androidx.compose.ui.test.junit4)
    androidTestImplementation(libs.androidx.espresso.core)
    androidTestImplementation(libs.androidx.junit)
    debugImplementation(libs.androidx.compose.ui.test.manifest)
    debugImplementation(libs.androidx.compose.ui.tooling)
}