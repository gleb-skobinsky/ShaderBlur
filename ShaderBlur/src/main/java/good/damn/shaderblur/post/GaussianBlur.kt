package good.damn.shaderblur.post

import android.graphics.Bitmap
import android.graphics.Canvas
import android.opengl.GLES20.GL_CLAMP_TO_EDGE
import android.opengl.GLES20.GL_COLOR_ATTACHMENT0
import android.opengl.GLES20.GL_COLOR_BUFFER_BIT
import android.opengl.GLES20.GL_DEPTH_ATTACHMENT
import android.opengl.GLES20.GL_DEPTH_BUFFER_BIT
import android.opengl.GLES20.GL_FLOAT
import android.opengl.GLES20.GL_FRAGMENT_SHADER
import android.opengl.GLES20.GL_FRAMEBUFFER
import android.opengl.GLES20.GL_FRAMEBUFFER_COMPLETE
import android.opengl.GLES20.GL_LINEAR
import android.opengl.GLES20.GL_RENDERBUFFER
import android.opengl.GLES20.GL_RGBA
import android.opengl.GLES20.GL_TEXTURE_2D
import android.opengl.GLES20.GL_TEXTURE_MAG_FILTER
import android.opengl.GLES20.GL_TEXTURE_MIN_FILTER
import android.opengl.GLES20.GL_TEXTURE_WRAP_S
import android.opengl.GLES20.GL_TEXTURE_WRAP_T
import android.opengl.GLES20.GL_TRIANGLES
import android.opengl.GLES20.GL_UNSIGNED_BYTE
import android.opengl.GLES20.GL_UNSIGNED_SHORT
import android.opengl.GLES20.GL_VERTEX_SHADER
import android.opengl.GLES20.glActiveTexture
import android.opengl.GLES20.glAttachShader
import android.opengl.GLES20.glBindTexture
import android.opengl.GLES20.glCheckFramebufferStatus
import android.opengl.GLES20.glClear
import android.opengl.GLES20.glClearColor
import android.opengl.GLES20.glCreateProgram
import android.opengl.GLES20.glDeleteFramebuffers
import android.opengl.GLES20.glDeleteProgram
import android.opengl.GLES20.glDeleteRenderbuffers
import android.opengl.GLES20.glDeleteTextures
import android.opengl.GLES20.glDisableVertexAttribArray
import android.opengl.GLES20.glDrawElements
import android.opengl.GLES20.glEnableVertexAttribArray
import android.opengl.GLES20.glFramebufferRenderbuffer
import android.opengl.GLES20.glFramebufferTexture2D
import android.opengl.GLES20.glGenFramebuffers
import android.opengl.GLES20.glGenRenderbuffers
import android.opengl.GLES20.glGenTextures
import android.opengl.GLES20.glGenerateMipmap
import android.opengl.GLES20.glGetAttribLocation
import android.opengl.GLES20.glGetUniformLocation
import android.opengl.GLES20.glLinkProgram
import android.opengl.GLES20.glTexImage2D
import android.opengl.GLES20.glTexParameteri
import android.opengl.GLES20.glUniform1f
import android.opengl.GLES20.glUniform1i
import android.opengl.GLES20.glUniform2f
import android.opengl.GLES20.glUseProgram
import android.opengl.GLES20.glVertexAttribPointer
import android.opengl.GLES20.glViewport
import android.opengl.GLUtils.texImage2D
import android.view.View
import good.damn.shaderblur.opengl.OpenGLUtils
import org.intellij.lang.annotations.Language
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer

@Language("GLSL")
private const val FragmentShader =
    """
        precision mediump float;

        uniform sampler2D input_texture;
        uniform vec2 texture_resolution;
        uniform float resFactor;
        const float radius = 7.0;
        
        void main() {
            
            // Calculate downsampled texture coordinates
            vec2 uv = gl_FragCoord.xy / texture_resolution;
            
            // Flip the y-coordinate
            uv.y = 1.0 - uv.y;
            
            // Calculate step size for downsampling
            vec2 stepsize = 1.0 / texture_resolution;
            
            // Initialize sum and weight for blur calculation
            vec4 sum = vec4(0.0);
            float weight = 0.0;
            
            // Iterate over a smaller region for downsampling
            for (float i = -radius; i <= radius; i++) { 
                for (float j = -radius; j <= radius; j++) {
                    // Calculate offset in downsampled texture space
                    vec2 offset = vec2(i, j) * stepsize; 
                    
                    // Sample from the input texture using the downsampled coordinates
                    vec4 color = texture2D(input_texture, uv + offset);
                    
                    // Calculate distance and weight
                    float distance = length(offset); 
                    float w = exp(-(distance * distance) / 2.0);
                    
                    // Accumulate sum and weight
                    sum += color * w; 
                    weight += w;
                }
            }
            
            // Calculate blurred color
            vec4 blurredColor = sum / weight; 
            
            // Output the blurred color
            gl_FragColor = blurredColor; 
        }
    """

@Language("GLSL")
private const val VertexShader = """
        attribute vec4 position;
        
        void main() {
            gl_Position = position;
        }
        """


class GaussianBlur(
    private val targetView: View
) {
    private var texture: IntArray? = null

    private var scaleFactor = 0.1f
        set(value) {
            field = value
            scaledWidth = width * scaleFactor
            scaledHeight = height * scaleFactor
        }

    private var scaledWidth = 0f
    private var scaledHeight = 0f

    private var width = 0f
    private var height = 0f

    private lateinit var vertexBuffer: FloatBuffer
    private lateinit var indicesBuffer: ShortBuffer

    private var verticalProgram = 0
    private var nativeProgram = 0

    private val canvas = Canvas()

    private var inputBitmap: Bitmap? = null

    private var depthBuffer: IntArray? = null
    private var frameBuffer: IntArray? = null

    fun create(
        vertexBuffer: FloatBuffer,
        indicesBuffer: ShortBuffer
    ) {
        this.vertexBuffer = vertexBuffer
        this.indicesBuffer = indicesBuffer

        nativeProgram = glCreateProgram()
        glAttachShader(
            nativeProgram,
            OpenGLUtils.loadShader(GL_VERTEX_SHADER, VertexShader)
        )
        glAttachShader(
            nativeProgram,
            OpenGLUtils.loadShader(GL_FRAGMENT_SHADER, FragmentShader)
        )
        glLinkProgram(nativeProgram)
    }

    fun layout(
        width: Int,
        height: Int
    ) {
        deleteFBO()
        this.width = width.toFloat()
        this.height = height.toFloat()

        scaledWidth = this.width * scaleFactor
        scaledHeight = this.height * scaleFactor
        inputBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        texture = intArrayOf(1)
        frameBuffer = intArrayOf(1)
        depthBuffer = intArrayOf(1)
        glGenFramebuffers(1, frameBuffer, 0)
        glGenRenderbuffers(1, depthBuffer, 0)

        configTexture(texture)
    }

    fun clean() {
        if (verticalProgram != 0) glDeleteProgram(verticalProgram)
        if (nativeProgram != 0) glDeleteProgram(nativeProgram)
        deleteFBO()
    }

    fun draw() {
        drawBitmap()
        drawBlur()
    }

    private fun deleteFBO() {
        inputBitmap?.let { if (!it.isRecycled) it.recycle() }
        frameBuffer?.let { glDeleteFramebuffers(1, it, 0) }
        texture?.let { glDeleteTextures(1, it, 0) }
        depthBuffer?.let { glDeleteRenderbuffers(1, it, 0) }
    }

    private fun drawBlur() {
        glUseProgram(nativeProgram)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture!![0], 0)
        glFramebufferRenderbuffer(
            GL_FRAMEBUFFER,
            GL_DEPTH_ATTACHMENT,
            GL_RENDERBUFFER,
            depthBuffer!![0]
        )
        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) return
        glActiveTexture(texture!![0])
        glBindTexture(GL_TEXTURE_2D, texture!![0])
        texImage2D(GL_TEXTURE_2D, 0, inputBitmap, 0)
        // bind the texture and generate the mipmap
        glGenerateMipmap(GL_TEXTURE_2D)
        // set the view port to the full width and height of the view
        glViewport(0, 0, width.toInt(), height.toInt())
        // transparent clear color
        glClearColor(0f, 0f, 0f, 0f)
        glClear(GL_DEPTH_BUFFER_BIT or GL_COLOR_BUFFER_BIT)
        // handle positioning
        val positionHandle = glGetAttribLocation(nativeProgram, "position")
        glEnableVertexAttribArray(positionHandle)
        glVertexAttribPointer(positionHandle, 2, GL_FLOAT, false, 8, vertexBuffer)
        // set the uniforms
        glUniform1i(glGetUniformLocation(nativeProgram, "input_texture"), 0)
        glUniform1f(glGetUniformLocation(verticalProgram, "resFactor"), scaleFactor)
        glUniform2f(glGetUniformLocation(nativeProgram, "texture_resolution"), width, height)
        glDrawElements(GL_TRIANGLES, indicesBuffer.capacity(), GL_UNSIGNED_SHORT, indicesBuffer)
        glDisableVertexAttribArray(positionHandle)
    }

    private fun configTexture(texture: IntArray?) {
        glGenTextures(1, texture, 0)
        glBindTexture(GL_TEXTURE_2D, texture!![0])
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        val (w, h) = (scaledWidth.toInt() to scaledHeight.toInt())
        val buf = IntArray(w * h)
        val mTexBuffer = ByteBuffer.allocateDirect(buf.size * Float.SIZE_BYTES)
            .order(ByteOrder.nativeOrder()).asIntBuffer()

        glTexImage2D(
            GL_TEXTURE_2D,
            0, GL_RGBA, w, h, 0,
            GL_RGBA, GL_UNSIGNED_BYTE, mTexBuffer
        )
    }

    private fun drawBitmap() {
        try {
            val rawOffset = -targetView.scrollY.toFloat()
            canvas.setBitmap(inputBitmap)
            canvas.translate(
                0f,
                rawOffset.coerceAtMost(0f)
            )
            targetView.draw(canvas)
        } catch (_: Exception) {
        }
    }
}