package com.example.edgeai

import android.graphics.Bitmap
import android.graphics.RectF
import kotlin.math.max

fun Bitmap.cropFace(box: RectF, scale: Float = 1f): Bitmap {
    // Make the crop square, centred on the face, with margin (FairFace crops include some context)
    val side = max(box.width(), box.height()) * scale
    val left = (box.centerX() - side / 2).toInt().coerceIn(0, width - 1)
    val top = (box.centerY() - side / 2).toInt().coerceIn(0, height - 1)
    val right = (box.centerX() + side / 2).toInt().coerceIn(left + 1, width)
    val bottom = (box.centerY() + side / 2).toInt().coerceIn(top + 1, height)
    return Bitmap.createBitmap(this, left, top, right - left, bottom - top)
}/* no-op */