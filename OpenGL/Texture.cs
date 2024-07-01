using NLog;
using Silk.NET.OpenGL;
using StbImageSharp;

namespace OpenCL_Barnes_Hut.OpenGL;

public class Texture : IDisposable {
	private static readonly Logger logger = LogManager.GetCurrentClassLogger();
	private readonly GL gl;
	private readonly uint handle;

	public unsafe Texture(GL openGl, string path) {
		logger.Debug("Loading new texture.");

		gl = openGl;
		handle = gl.GenTexture();
		Bind();

		var imageResult = ImageResult.FromMemory(File.ReadAllBytes(path), ColorComponents.RedGreenBlueAlpha);

		fixed (byte* data = imageResult.Data) {
			gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba, (uint)imageResult.Width,
				(uint)imageResult.Height, 0, PixelFormat.Rgba, PixelType.UnsignedByte, data);
			SetParams();
		}
	}

	public unsafe Texture(GL openGl, Span<byte> data, uint width, uint height) {
		logger.Debug("Loading new texture.");

		gl = openGl;
		handle = gl.GenTexture();
		Bind();

		fixed (void* pData = &data[0]) {
			gl.TexImage2D(TextureTarget.Texture2D, 0, InternalFormat.Rgba, width, height, 0, PixelFormat.Rgba,
				PixelType.UnsignedByte, pData);
			SetParams();
		}
	}

	public void Dispose() {
		gl.DeleteTexture(handle);
	}

	private void SetParams() {
		logger.Debug("Setting texture parameters.");
		gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapS, (int)GLEnum.ClampToEdge);
		gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureWrapT, (int)GLEnum.ClampToEdge);
		gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMinFilter, (int)GLEnum.LinearMipmapLinear);
		gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMagFilter, (int)GLEnum.Linear);
		gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureBaseLevel, 0);
		gl.TexParameter(TextureTarget.Texture2D, TextureParameterName.TextureMaxLevel, 8);
		gl.GenerateMipmap(TextureTarget.Texture2D);
	}

	public void Bind(TextureUnit textureSlot = TextureUnit.Texture0) {
		gl.ActiveTexture(textureSlot);
		gl.BindTexture(TextureTarget.Texture2D, handle);
	}
}