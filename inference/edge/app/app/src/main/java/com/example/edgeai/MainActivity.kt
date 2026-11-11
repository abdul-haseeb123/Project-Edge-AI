package com.example.edgeai

import android.Manifest
import android.content.ContentValues
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.icu.text.SimpleDateFormat
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.ActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageProxy
import androidx.camera.view.CameraController
import androidx.camera.view.LifecycleCameraController
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import coil3.ImageLoader
import coil3.request.ImageRequest
import coil3.request.ImageResult
import coil3.request.SuccessResult
import coil3.request.allowHardware
import coil3.toBitmap
import com.example.edgeai.presentation.CameraPreview
import com.example.edgeai.presentation.FaceDetectorHelper
import com.example.edgeai.ui.theme.EdgeAITheme
import com.google.mediapipe.tasks.vision.core.RunningMode
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.Locale
import java.util.concurrent.Executors
import kotlin.collections.map
import kotlin.collections.orEmpty
import kotlin.math.max

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        if (!hasRequiredPermissions()) {
            ActivityCompat.requestPermissions(this, CAMERAX_PERMISSIONS, 0)
        }
        setContent {
            EdgeAITheme {
                // External mutable state the listener writes into

                val scope = rememberCoroutineScope()


                val imageLoader = remember {
                    ImageLoader(context = applicationContext)
                }


                var latestResult by remember {
                    mutableStateOf<FaceDetectorHelper.ResultBundle?>(null)
                }
                var genderLabels by remember { mutableStateOf<List<String>>(emptyList()) }
                var ageLabels by remember { mutableStateOf<List<String>>(emptyList()) }
                var expressionLabels by remember { mutableStateOf<List<String>>(emptyList()) }

                val genderClassifier = remember {
                    UltralyticsYOLOClassifier(
                        applicationContext,
                        "gender-cls.tflite",
                        "gender-labels.txt"
                    )
                }
                val ageClassifier = remember {
                    UltralyticsYOLOClassifier(
                        applicationContext,
                        "age_finetuned-cls.tflite",
                        "age_finetuned-labels.txt"
                    )
                }

                val expressionClassifier = remember {
                    UltralyticsYOLOClassifier(
                        applicationContext,
                        "expression_finetuned-cls.tflite",
                        "expression_finetuned-labels.txt"
                    )
                }


                var errorMessage by remember { mutableStateOf<String?>(null) }

                val inferenceExecutor = remember { Executors.newSingleThreadExecutor() }
                val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
                var busy by remember { mutableStateOf(false) }

                val photoAnalyzer = remember {
                    FaceDetectorHelper(
                        context = applicationContext,
                        runningMode = RunningMode.IMAGE
                    )
                }

                val analyzer = remember {
                    FaceDetectorHelper(
                        context = applicationContext,
                        runningMode = RunningMode.LIVE_STREAM,
                        faceDetectorListener = object : FaceDetectorHelper.DetectorListener {
                            override fun onResults(resultBundle: FaceDetectorHelper.ResultBundle) {
                                latestResult = resultBundle

                                val frame = resultBundle.frame ?: return
                                val detections =
                                    resultBundle.results.firstOrNull()?.detections().orEmpty()
                                if (detections.isEmpty()) {
                                    genderLabels = emptyList()
                                    ageLabels = emptyList()
                                    expressionLabels = emptyList()
                                    return

                                }

                                if (busy) return   // classifier busy: skip

                                val crops = detections.map { frame.cropFace(it.boundingBox()) }
                                busy = true
                                inferenceExecutor.execute {
                                    try {
                                        genderLabels =
                                            crops.map { c -> c.let(genderClassifier::classify) }
                                        ageLabels =
                                            crops.map { c -> c.let(ageClassifier::classify) }
                                        expressionLabels =
                                            crops.map { c -> c.let(expressionClassifier::classify) }

                                    } finally {
                                        busy = false
                                    }
                                }
                            }

                            override fun onError(error: String, errorCode: Int) {
                                errorMessage = error
                            }
                        }
                    )
                }

                val pickMedia = rememberLauncherForActivityResult(
                    contract = ActivityResultContracts.PickVisualMedia(),
                    onResult = { uri ->
                        scope.launch {
                            try {
                                // Step A: Load the Uri to a Software-backed Bitmap
                                val request = ImageRequest.Builder(applicationContext)
                                    .data(uri)
                                    .allowHardware(false) // MANDATORY: Allows pixel manipulation & editing
                                    .build()

                                val result = imageLoader.execute(request)
                                if (result !is SuccessResult) return@launch

                                val originalBitmap = result.image.toBitmap()

                                Log.d("ImgUri", "Image received successfully")

                                // Heavy work off the main thread, one image at a time
                                onTakePhoto(
                                    bitmap = originalBitmap,
                                    analyzer = photoAnalyzer,
                                    genderClassifier = genderClassifier,
                                    ageClassifier = ageClassifier,
                                    expressionClassifier = expressionClassifier
                                )


                                true
                            } catch (e: Exception) {
                                e.printStackTrace()
                                false
                            }

                        }
                    }
                )


                val controller = remember {
                    LifecycleCameraController(applicationContext).apply {
                        setEnabledUseCases(
                            CameraController.IMAGE_CAPTURE or
                                    CameraController.IMAGE_ANALYSIS
                        )
                        cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
                        imageAnalysisOutputImageFormat = ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
                        setImageAnalysisAnalyzer(
                            cameraExecutor,
                            analyzer
                        )
                    }
                }

                DisposableEffect(Unit) {
                    onDispose {
                        analyzer.clearFaceDetector()
                        cameraExecutor.shutdown()
                        inferenceExecutor.execute {
                            genderClassifier.close()
                            ageClassifier.close()
                            expressionClassifier.close()
                        }
                        inferenceExecutor.shutdown()
                    }
                }

                Scaffold { innerPadding ->
                    Box(Modifier
                        .padding(innerPadding)
                        .fillMaxSize()) {
                        CameraPreview(controller, Modifier.fillMaxSize())
                        FaceOverlay(
                            latestResult,
                            genderLabels,
                            ageLabels,
                            expressionLabels,
                            Modifier.fillMaxSize()
                        )

                        val faces = latestResult?.results?.firstOrNull()?.detections()?.size ?: 0
                        Text(
                            text = errorMessage
                                ?: if (faces > 0) "$faces faces detected" else "No faces detected",
                            modifier = Modifier
                                .fillMaxWidth()
                                .align(Alignment.TopCenter)
                                .background(MaterialTheme.colorScheme.primaryContainer)
                                .padding(8.dp),
                            textAlign = TextAlign.Center,
                            fontSize = 20.sp,
                            color = MaterialTheme.colorScheme.primary
                        )

                        IconButton(
                            onClick = {
                                if (controller.cameraSelector == CameraSelector.DEFAULT_FRONT_CAMERA) {
                                    analyzer.isFrontCamera = false
                                    photoAnalyzer.isFrontCamera = false
                                    controller.cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                                } else {
                                    analyzer.isFrontCamera = true
                                    photoAnalyzer.isFrontCamera = true
                                    controller.cameraSelector = CameraSelector.DEFAULT_FRONT_CAMERA
                                }
                            },
                            modifier = Modifier.align(Alignment.TopStart)
                        ) {
                            Icon(
                                painter = painterResource(R.drawable.cameraswitch_24px),
                                contentDescription = "camera switch button"
                            )
                        }

                        Row(
                            modifier = Modifier
                                .align(Alignment.BottomCenter)
                                .padding(8.dp)
                        ) {
                            IconButton(
                                onClick = {
                                    pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))


                                },
                            ) {
                                Icon(
                                    painter = painterResource(R.drawable.photo_library_40px),
                                    contentDescription = "Camera capture button"
                                )
                            }


                            IconButton(
                                onClick = {
                                    takePhoto(
                                        controller,
                                        photoAnalyzer,
                                        genderClassifier,
                                        ageClassifier,
                                        expressionClassifier
                                    )
                                },
                            ) {
                                Icon(
                                    painter = painterResource(R.drawable.camera_40px),
                                    contentDescription = "Camera capture button"
                                )
                            }
                        }

                    }
                }
            }
        }
    }


    private fun savePhoto(bitmap: Bitmap) {
        var imgUri: Uri
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            imgUri = MediaStore.Images.Media.getContentUri(
                MediaStore.VOLUME_EXTERNAL_PRIMARY
            )
        } else {
            imgUri = MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        }

        val name =
            SimpleDateFormat("dd-MM-yyyy HH-mm-ss", Locale.US).format(System.currentTimeMillis())

        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")

            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/EdgeAI")
                put(MediaStore.Images.Media.IS_PENDING, 1)
            }
        }


        imgUri = contentResolver.insert(imgUri, contentValues) ?: run {
            Log.e("SavePhoto", "MediaStore insert failed")
            return
        }

        val outputStream = contentResolver.openOutputStream(imgUri)
        if (outputStream != null) {
            bitmap.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
        }

        outputStream?.close()


        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            contentValues.clear()
            contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
            contentResolver.update(imgUri, contentValues, null, null)
        }

        Toast.makeText(applicationContext, imgUri.toString(), Toast.LENGTH_SHORT).show()

        return
    }


    private fun onTakePhoto(
        image: ImageProxy? = null,
        bitmap: Bitmap? = null,
        analyzer: FaceDetectorHelper,
        genderClassifier: UltralyticsYOLOClassifier,
        ageClassifier: UltralyticsYOLOClassifier,
        expressionClassifier: UltralyticsYOLOClassifier
    ) {
        if (image == null && bitmap == null) return
        if (image != null && bitmap != null) return

        val resultBundle: FaceDetectorHelper.ResultBundle = if (image != null) {
            analyzer.detectImage(image) ?: return
        } else if (bitmap != null) {
            analyzer.detectBitmap(bitmap) ?: return
        } else {
            return
        }
        val frame = resultBundle.frame
        if (frame == null) {
            Log.w("OnTakePhoto", "resultBundle.frame is null"); return
        }
        val detections = resultBundle.results.firstOrNull()?.detections().orEmpty()
        if (detections.isEmpty()) return

        val classifyStart = SystemClock.elapsedRealtime()

        // One label string per face
        val labels = detections.map { d ->
            val crop = frame.cropFace(d.boundingBox())
            "${genderClassifier.classify(crop)} · ${ageClassifier.classify(crop)} · ${
                expressionClassifier.classify(
                    crop
                )
            }"
        }

        val classifyMs = SystemClock.elapsedRealtime() - classifyStart

        val output = frame.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(output)

        // ---- Tuning knobs ----
        val longSide = max(output.width, output.height).toFloat()
        val strokeRatio = 0.005f      // box line = 0.5% of the long side
        val fitFraction = 1.0f        // label width as a fraction of box width (0.8 = smaller text)
        val padRatio = 0.3f           // padding as a fraction of text size
        val minText = longSide * 0.012f   // floor so tiny faces stay readable
        val maxText = longSide * 0.035f   // cap so lone big faces don't get huge text
// ----------------------

        val boxColor = ContextCompat.getColor(applicationContext, R.color.teal_700)
        val strokeW = (longSide * strokeRatio).coerceAtLeast(4f)

        val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = boxColor
            style = Paint.Style.STROKE
            strokeWidth = strokeW
            strokeJoin = Paint.Join.ROUND
        }
        val bgPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = boxColor
            style = Paint.Style.FILL
        }
        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = ContextCompat.getColor(applicationContext, R.color.white)
            style = Paint.Style.FILL
            isFakeBoldText = true
        }

        detections.forEachIndexed { i, d ->
            val box = d.boundingBox()
            canvas.drawRect(box, boxPaint)

            val label = labels[i]

            // Measure the label at a reference size, then solve for the size that fits the box width:
            // labelWidth = textSize * (refWidth / 100) + 2 * padRatio * textSize
            textPaint.textSize = 100f
            val refWidth = textPaint.measureText(label)
            val fitted = (box.width() * fitFraction) / (refWidth / 100f + 2 * padRatio)
            textPaint.textSize = fitted.coerceIn(minText, maxText)

            val pad = textPaint.textSize * padRatio
            val fm = textPaint.fontMetrics
            val bgW = textPaint.measureText(label) + pad * 2
            val bgH = (fm.descent - fm.ascent) + pad * 2

            // Keep the label inside the photo horizontally; overlap the stroke so there is no gap
            val bgLeft = box.left.coerceIn(0f, (output.width - bgW).coerceAtLeast(0f))
            var bgTop = box.top + strokeW / 2f - bgH
            if (bgTop < 0f) bgTop =
                box.top - strokeW / 2f      // no room above: place inside the box

            val r = pad * 0.6f
            canvas.drawRoundRect(RectF(bgLeft, bgTop, bgLeft + bgW, bgTop + bgH), r, r, bgPaint)
            canvas.drawText(label, bgLeft + pad, bgTop + pad - fm.ascent, textPaint)
        }

        // ---- Inference time box (top-left corner) ----
        val detectMs = resultBundle.inferenceTime
        val lines = listOf(
            "Detect: $detectMs ms",
            "Classify: $classifyMs ms",
            "Total: ${detectMs + classifyMs} ms"
        )

        val infoText = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = ContextCompat.getColor(applicationContext, R.color.white)
            textSize = longSide * 0.025f          // fixed size, independent of faces
            isFakeBoldText = true
        }
        val infoBg = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = boxColor
            style = Paint.Style.FILL
        }

        val margin = longSide * 0.02f
        val pad = infoText.textSize * 0.5f
        val fm = infoText.fontMetrics
        val lineH = fm.descent - fm.ascent
        val textW = lines.maxOf { infoText.measureText(it) }

        val rect = RectF(
            margin,
            margin,
            margin + textW + pad * 2,
            margin + lineH * lines.size + pad * 2
        )
        canvas.drawRoundRect(rect, pad * 0.6f, pad * 0.6f, infoBg)

        lines.forEachIndexed { i, line ->
            canvas.drawText(line, rect.left + pad, rect.top + pad - fm.ascent + i * lineH, infoText)
        }

        savePhoto(output)
    }


    private fun takePhoto(
        controller: LifecycleCameraController,
        analyzer: FaceDetectorHelper,
        genderClassifier: UltralyticsYOLOClassifier,
        ageClassifier: UltralyticsYOLOClassifier,
        expressionClassifier: UltralyticsYOLOClassifier
    ) {
        controller.takePicture(
            ContextCompat.getMainExecutor(applicationContext),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(image: ImageProxy) {
                    onTakePhoto(
                        image, analyzer = analyzer,
                        genderClassifier = genderClassifier,
                        ageClassifier = ageClassifier,
                        expressionClassifier = expressionClassifier
                    )
                    image.close()
                }
            }
        )
    }

    private fun hasRequiredPermissions(): Boolean {
        return CAMERAX_PERMISSIONS.all {
            ContextCompat.checkSelfPermission(
                applicationContext,
                it
            ) == PackageManager.PERMISSION_GRANTED
        }
    }


    companion object {
        private val CAMERAX_PERMISSIONS = arrayOf(
            Manifest.permission.CAMERA
        )
    }
}


@Composable
fun FaceOverlay(
    bundle: FaceDetectorHelper.ResultBundle?,
    genderLabels: List<String>,
    ageLabels: List<String>,
    expressionLabels: List<String>,
    modifier: Modifier = Modifier
) {
    val boxColor = MaterialTheme.colorScheme.primary
    val textMeasurer = rememberTextMeasurer()

    Canvas(modifier = modifier) {
        val b = bundle ?: return@Canvas
        val imgW = b.inputImageWidth.toFloat()
        val imgH = b.inputImageHeight.toFloat()
        val scale = max(size.width / imgW, size.height / imgH)   // FILL_CENTER math
        val offsetX = (size.width - imgW * scale) / 2f
        val offsetY = (size.height - imgH * scale) / 2f

        val detections = b.results.firstOrNull()?.detections().orEmpty()
        detections.forEachIndexed { i, d ->
            val box = d.boundingBox()
            val topLeft = Offset(box.left * scale + offsetX, box.top * scale + offsetY)
            drawRect(
                color = boxColor,
                topLeft = topLeft,
                size = Size(box.width() * scale, box.height() * scale),
                style = Stroke(width = 4.dp.toPx())
            )
            if (genderLabels.size == detections.size && ageLabels.size == detections.size && expressionLabels.size == detections.size) {
                drawText(
                    textMeasurer = textMeasurer,
                    text = "${genderLabels[i]} ${ageLabels[i]} ${expressionLabels[i]}",
                    topLeft = Offset(topLeft.x, (topLeft.y - 24.dp.toPx()).coerceAtLeast(0f)),
                    style = TextStyle(color = Color.White, fontSize = 14.sp, background = boxColor)
                )
            }
        }
    }
}