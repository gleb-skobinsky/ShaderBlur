package good.damn.shaderblur.opengl

import android.opengl.GLES20.GL_COLOR_BUFFER_BIT
import android.opengl.GLES20.GL_DEPTH_BUFFER_BIT
import android.opengl.GLSurfaceView
import android.view.View
import good.damn.shaderblur.post.GaussianBlur
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.ShortBuffer
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

class Renderer(targetView: View) : GLSurfaceView.Renderer {

    private var mOnFrameCompleteListener: Runnable? = null

    private lateinit var mVertexBuffer: FloatBuffer
    private lateinit var mIndicesBuffer: ShortBuffer

    private val mIndices: ShortArray = shortArrayOf(0, 1, 2, 0, 2, 3)

    private val mSquareCoords: FloatArray = floatArrayOf(
        -1.0f, 1.0f,  // top left
        -1.0f, -1.0f, // bottom left
        1.0f, -1.0f,  // bottom right
        1.0f, 1.0f,   // top right
    )

    private var mWidth = 1f
    private var mHeight = 1f

    private val mBlurEffect = GaussianBlur(targetView)

    override fun onSurfaceCreated(gl: GL10?, p1: EGLConfig?) {
        gl?.glClearColor(0.0f, 0.0f, 0.0f, 0.0f)

        val byteBuffer =
            ByteBuffer.allocateDirect(mSquareCoords.size * 4)
        byteBuffer.order(ByteOrder.nativeOrder())
        mVertexBuffer = byteBuffer.asFloatBuffer()
        mVertexBuffer.put(mSquareCoords)
        mVertexBuffer.position(0)

        val drawByteBuffer: ByteBuffer =
            ByteBuffer.allocateDirect(mIndices.size * 2)
        drawByteBuffer.order(ByteOrder.nativeOrder())
        mIndicesBuffer = drawByteBuffer.asShortBuffer()
        mIndicesBuffer.put(mIndices)
        mIndicesBuffer.position(0)

        mBlurEffect.create(
            mVertexBuffer,
            mIndicesBuffer
        )
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        gl?.glViewport(0, 0, width, height)
        mWidth = width.toFloat()
        mHeight = height.toFloat()
        mBlurEffect.layout(width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        gl?.glClear(GL_COLOR_BUFFER_BIT or GL_DEPTH_BUFFER_BIT)
        mBlurEffect.draw()
        mOnFrameCompleteListener?.run()
    }

    fun clean() {
        mBlurEffect.clean()
    }

}