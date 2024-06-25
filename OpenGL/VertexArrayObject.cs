using NLog;
using Silk.NET.OpenGL;

namespace OpenCL_Barnes_Hut.OpenGL;

public class VertexArrayObject<TVertexType, TIndexType> : IDisposable
	where TIndexType : unmanaged where TVertexType : unmanaged {
	private static readonly Logger logger = LogManager.GetCurrentClassLogger();

	private readonly GL gl;
	private readonly uint handle;

	public VertexArrayObject(GL openGl, BufferObject<TVertexType> vbo, BufferObject<TIndexType> ebo) {
		gl = openGl;
		handle = gl.GenVertexArray();

		Bind();

		vbo.Bind();
		ebo.Bind();
	}

	public void Dispose() {
		gl.DeleteVertexArray(handle);
	}

	public unsafe void VertexAttributePointer(uint index, int count, VertexAttribPointerType type, uint vertexSize,
		int offSet) {
		gl.VertexAttribPointer(index, count, type, false, vertexSize * (uint)sizeof(TVertexType),
			(void*)(offSet * sizeof(TVertexType)));
		gl.EnableVertexAttribArray(index);
	}

	public void Bind() {
		gl.BindVertexArray(handle);
	}
}