using NLog;
using Silk.NET.OpenGL;

namespace OpenCL_Barnes_Hut.OpenGL;

public class BufferObject<TDataType> : IDisposable where TDataType : unmanaged {
	private static readonly Logger logger = LogManager.GetCurrentClassLogger();
	private readonly BufferTargetARB bufferType;
	private readonly GL gl;
	private readonly uint handle;

	public unsafe BufferObject(GL openGl, Span<TDataType> data, BufferTargetARB bufferTypeIn) {
		gl = openGl;
		bufferType = bufferTypeIn;

		handle = gl.GenBuffer();

		Bind();

		fixed (void* pData = data) {
			gl.BufferData(bufferType, (nuint)(data.Length * sizeof(TDataType)), pData, BufferUsageARB.StaticDraw);
		}
	}

	public void Dispose() {
		gl.DeleteBuffer(handle);
	}

	public void Bind() {
		gl.BindBuffer(bufferType, handle);
	}
}