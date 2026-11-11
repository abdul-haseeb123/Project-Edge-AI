package com.example.edgeai

import android.content.Context
import android.graphics.Bitmap
import androidx.core.graphics.scale
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel


class UltralyticsYOLOClassifier(
    val context: Context,
    modelAsset: String,
    val labelsAsset: String
) {

    private var labels: List<String> = emptyList()
    private val size = 112                           // image size from your metadata

    // 1. Load the model from assets, CPU execution
    private val model = CompiledModel.create(
        context.assets,
        modelAsset,
        CompiledModel.Options(Accelerator.CPU)
    )

    // 2. Allocate input/output buffers once and reuse them every call
    private val inputBuffers = model.createInputBuffers()
    private val outputBuffers = model.createOutputBuffers()

    // reused scratch arrays, so no allocation per frame
    private val pixels = IntArray(size * size)
    private val floats = FloatArray(size * size * 3)

    init {
        labels = loadLabels()
    }

    fun classify(crop: Bitmap): String {
        // 3. Resize the crop to 112x112
        val resized = crop.scale(size, size, true)
        resized.getPixels(pixels, 0, size, 0, 0, size, size)

        // 4. NHWC, RGB, scaled to 0..1 (Ultralytics preprocessing)

        var rIdx = 0
        var gIdx = size * size
        var bIdx = size * size * 2

        for (p in pixels) {
            floats[rIdx++] = ((p shr 16) and 0xFF) / 255f   // R
            floats[gIdx++] = ((p shr 8) and 0xFF) / 255f    // G
            floats[bIdx++] = (p and 0xFF) / 255f            // B
        }

        inputBuffers[0].writeFloat(floats)

        // 5. Run inference
        model.run(inputBuffers, outputBuffers)

        // 6. Read the 2 class probabilities
        val probs = outputBuffers[0].readFloat()
        val best = probs.indices.maxByOrNull { probs[it] } ?: 0

        val res = "${labels[best]} ${"%.2f".format(probs[best])}"

        return res

    }

    private fun loadLabels(): List<String> {
        return context.assets.open(labelsAsset).bufferedReader()
            .useLines { lines ->
                lines.filter { it.isNotBlank() }.toList()
            }
    }


    fun close() {
        inputBuffers.forEach { it.close() }
        outputBuffers.forEach { it.close() }
        model.close()
    }

}