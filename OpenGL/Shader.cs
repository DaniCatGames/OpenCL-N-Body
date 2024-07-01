using System.Numerics;
using NLog;
using Silk.NET.OpenGL;

namespace OpenCL_Barnes_Hut.OpenGL;

public class Shader : IDisposable {
	private static readonly Logger logger = LogManager.GetCurrentClassLogger();
	private readonly GL gl;
	private readonly uint handle;

	public Shader(GL openGl, string vertPath, string fragPath) {
		logger.Info("Creating new Shader.");
		gl = openGl;

		var vertex = LoadShader(ShaderType.VertexShader, vertPath);
		var fragment = LoadShader(ShaderType.FragmentShader, fragPath);

		handle = gl.CreateProgram();

		gl.AttachShader(handle, vertex);
		gl.AttachShader(handle, fragment);
		gl.LinkProgram(handle);

		gl.GetProgram(handle, GLEnum.LinkStatus, out var status);
		if (status == 0) logger.Error($"Program failed to link with error: {gl.GetProgramInfoLog(handle)}");

		gl.DetachShader(handle, vertex);
		gl.DetachShader(handle, fragment);
		gl.DeleteShader(vertex);
		gl.DeleteShader(fragment);
	}

	public void Dispose() {
		gl.DeleteProgram(handle);
	}

	public void Use() {
		gl.UseProgram(handle);
	}

	public void SetUniform(string name, int value) {
		logger.Debug($"Setting uniform {name}.");
		var location = gl.GetUniformLocation(handle, name);
		if (location == -1) logger.Error($"{name} uniform not found on shader.");
		gl.Uniform1(location, value);
	}

	public void SetUniform(string name, float value) {
		logger.Debug($"Setting uniform {name}.");
		var location = gl.GetUniformLocation(handle, name);
		if (location == -1) logger.Error($"{name} uniform not found on shader.");
		gl.Uniform1(location, value);
	}

	public void SetUniform(string name, Vector2 value) {
		logger.Debug($"Setting uniform {name}.");
		var location = gl.GetUniformLocation(handle, name);
		if (location == -1) logger.Error($"{name} uniform not found on shader.");
		gl.Uniform2(location, value);
	}

	public unsafe void SetUniform(string name, Matrix4x4 value) {
		logger.Debug($"Setting uniform {name}.");
		var location = gl.GetUniformLocation(handle, name);
		if (location == -1) logger.Error($"{name} uniform not found on shader.");
		gl.UniformMatrix4(location, 1, false, (float*)&value);
	}

	private uint LoadShader(ShaderType type, string path) {
		logger.Info("Loading shader.");
		var str = File.ReadAllText(path);
		var shaderHandle = gl.CreateShader(type);

		gl.ShaderSource(shaderHandle, str);
		gl.CompileShader(shaderHandle);

		var infoLog = gl.GetShaderInfoLog(handle);
		if (!string.IsNullOrWhiteSpace(infoLog))
			logger.Fatal($"Error compiling shader of type {type}, failed with error {infoLog}");

		return shaderHandle;
	}
}