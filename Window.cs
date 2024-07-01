using System.Numerics;
using NLog;
using OpenCL_Barnes_Hut.OpenGL;
using Silk.NET.Maths;
using Silk.NET.OpenGL;
using Silk.NET.Windowing;
using Shader = OpenCL_Barnes_Hut.OpenGL.Shader;
using Texture = OpenCL_Barnes_Hut.OpenGL.Texture;

namespace OpenCL_Barnes_Hut;

public class OpenClWindow {
	private static readonly Logger logger = LogManager.GetCurrentClassLogger();

	private static IWindow window;
	private static GL gl;

	private static BufferObject<float> Vbo;
	private static BufferObject<uint> Ebo;
	private static VertexArrayObject<float, uint> Vao;

	private static Texture Texture;

	private static Shader Shader;
	//private static readonly Transform[] Transforms = new Transform[4]; 3D

	private static readonly float[] Vertices = {
		//X    Y      Z     S    T
		1.0f, 1.0f, 0.0f, 1.0f, 0.0f,
		1.0f, -1.0f, 0.0f, 1.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, 1.0f, 1.0f, 0.0f, 0.0f
	};

	private static readonly uint[] Indices = {
		0, 1, 3,
		1, 2, 3
	};

	/* 3D
	private static Vector3 CameraPosition = new(0.0f, 0.0f, 3.0f);
	private static Vector3 CameraFront = new(0.0f, 0.0f, -1.0f);
	private static readonly Vector3 CameraUp = Vector3.UnitY;
	private static Vector3 CameraDirection = Vector3.Zero;
	private static float CameraYaw = -90f;
	private static float CameraPitch;
	private static float CameraZoom = 45f;

	private static Vector2 LastMousePosition;
	private static IKeyboard primaryKeyboard;*/


	public static void RunWindow() {
		var options = WindowOptions.Default with {
			Size = new Vector2D<int>(1920, 1080),
			Title = "OpenCL N-Body Simulation"
		};

		window = Window.Create(options);

		window.Load += OnLoad;
		window.Render += OnRender;
		window.FramebufferResize += OnFramebufferResize;
		window.Closing += OnClose;

		window.Run();

		window.Dispose();
	}

	private static void OnLoad() {
		logger.Info("Loading window.");

		/* 3D
		var input = window.CreateInput();
		primaryKeyboard = input.Keyboards.FirstOrDefault();
		if (primaryKeyboard != null) primaryKeyboard.KeyDown += KeyDown;
		for (var i = 0; i < input.Mice.Count; i++) {
			input.Mice[i].Cursor.CursorMode = CursorMode.Raw;
			input.Mice[i].MouseMove += OnMouseMove;
			input.Mice[i].Scroll += OnMouseWheel;
		}*/

		gl = GL.GetApi(window);

		Ebo = new BufferObject<uint>(gl, Indices, BufferTargetARB.ElementArrayBuffer);
		Vbo = new BufferObject<float>(gl, Vertices, BufferTargetARB.ArrayBuffer);
		Vao = new VertexArrayObject<float, uint>(gl, Vbo, Ebo);

		Vao.VertexAttributePointer(0, 3, VertexAttribPointerType.Float, 5, 0);
		Vao.VertexAttributePointer(1, 2, VertexAttribPointerType.Float, 5, 3);

		Shader = new Shader(gl, @"D:\Programming\C#\OpenCL Barnes-Hut\Shaders\shader.vert",
			@"D:\Programming\C#\OpenCL Barnes-Hut\Shaders\circleShader.frag");

		//Texture = new Texture(gl, @"D:\Programming\C#\OpenCL Barnes-Hut\OpenGL\Textures\cat.png");

		/* 3D
		//Transforms.

		//Translation
		Transforms[0] = new Transform();
		Transforms[0].Position = new Vector3(0.5f, 0.5f, 0f);

		//Rotation.
		Transforms[1] = new Transform();
		Transforms[1].Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, 1f);

		//Scaling.
		Transforms[2] = new Transform();
		Transforms[2].Scale = 0.5f;

		//Mixed transformation.
		Transforms[3] = new Transform();
		Transforms[3].Position = new Vector3(-0.5f, 0.5f, 0f);
		Transforms[3].Rotation = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, 1f);
		Transforms[3].Scale = 0.5f;*/
	}

	/* 3D
	private static void OnUpdate(double deltaTime) {
		var moveSpeed = 2.5f * (float)deltaTime;

		if (primaryKeyboard.IsKeyPressed(Key.W))
			//Move forwards
			CameraPosition += moveSpeed * CameraFront;
		if (primaryKeyboard.IsKeyPressed(Key.S))
			//Move backwards
			CameraPosition -= moveSpeed * CameraFront;
		if (primaryKeyboard.IsKeyPressed(Key.A))
			//Move left
			CameraPosition -= Vector3.Normalize(Vector3.Cross(CameraFront, CameraUp)) * moveSpeed;
		if (primaryKeyboard.IsKeyPressed(Key.D))
			//Move right
			CameraPosition += Vector3.Normalize(Vector3.Cross(CameraFront, CameraUp)) * moveSpeed;
	}*/

	private static unsafe void OnRender(double deltaTime) {
		//gl.Enable(EnableCap.DepthTest); 3D
		gl.Clear((uint)ClearBufferMask.ColorBufferBit /*| ClearBufferMask.DepthBufferBit*/);

		Vao.Bind();
		//Texture.Bind();

		/* 3D
		var difference = (float)(window.Time * 100);

		var size = window.FramebufferSize;

		var model = Matrix4x4.CreateRotationY(float.DegreesToRadians(difference)) *
		            Matrix4x4.CreateRotationX(float.DegreesToRadians(difference));
		var view = Matrix4x4.CreateLookAt(CameraPosition, CameraPosition + CameraFront, CameraUp);
		var projection = Matrix4x4.CreatePerspectiveFieldOfView(float.DegreesToRadians(CameraZoom),
			(float)size.X / size.Y, 0.1f, 100.0f);*/

		Shader.Use();
		//Shader.SetUniform("uTexture0", 0);
		/* 3D
		Shader.SetUniform("uModel", model);
		Shader.SetUniform("uView", view);
		Shader.SetUniform("uProjection", projection);*/
		Shader.SetUniform("uTime", (float)window.Time);
		Shader.SetUniform("uResolution", (Vector2)window.Size);

		gl.DrawElements(PrimitiveType.Triangles, (uint)Indices.Length, DrawElementsType.UnsignedInt, null);
	}

	private static void OnFramebufferResize(Vector2D<int> size) {
		gl.Viewport(size);
	}

	/* 3D
	 private static void OnMouseMove(IMouse mouse, Vector2 position) {
		var lookSensitivity = 0.1f;
		if (LastMousePosition == default) {
			LastMousePosition = position;
		}
		else {
			var xOffset = (position.X - LastMousePosition.X) * lookSensitivity;
			var yOffset = (position.Y - LastMousePosition.Y) * lookSensitivity;
			LastMousePosition = position;

			CameraYaw += xOffset;
			CameraPitch -= yOffset;

			CameraPitch = Math.Clamp(CameraPitch, -89.0f, 89.0f);

			CameraDirection.X = MathF.Cos(float.DegreesToRadians(CameraYaw)) *
			                    MathF.Cos(float.DegreesToRadians(CameraPitch));
			CameraDirection.Y = MathF.Sin(float.DegreesToRadians(CameraPitch));
			CameraDirection.Z = MathF.Sin(float.DegreesToRadians(CameraYaw)) *
			                    MathF.Cos(float.DegreesToRadians(CameraPitch));
			CameraFront = Vector3.Normalize(CameraDirection);
		}
	}

	private static void OnMouseWheel(IMouse mouse, ScrollWheel scrollWheel) {
		CameraZoom = Math.Clamp(CameraZoom - scrollWheel.Y, 1.0f, 45f);
	}

	private static void KeyDown(IKeyboard keyboard, Key key, int arg3) {
		if (key == Key.Escape) window.Close();
	}*/

	private static void OnClose() {
		logger.Info("Closing window.");
		Vbo.Dispose();
		Ebo.Dispose();
		Vao.Dispose();
		Shader.Dispose();
		//Texture.Dispose();
	}
}