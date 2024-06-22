using System.Text;
using Silk.NET.OpenCL;

namespace OpenCL_Barnes_Hut;

internal class Program {
	private const int NumberOfBodies = 2;
	private const int Iterations = 1;
	private const float DeltaTime = 100;

	private static unsafe void Main(string[] args) {
		var cl = CL.GetApi();
		nint context = 0;
		nint commandQueue = 0;
		nint program = 0;
		nint kernel = 0;
		nint device = 0;
		var writer = new StreamWriter(@"D:\Programming\OpenTK\OpenCL Barnes-Hut\NBodyData.txt");

		var memObjects = new nint[3];

		// Create an OpenCL context on first available platform
		context = CreateContext(cl);
		if (context == IntPtr.Zero) {
			Console.WriteLine("Failed to create OpenCL context.");
			return;
		}

		// Create a command-queue on the first device available
		// on the created context
		commandQueue = CreateCommandQueue(cl, context, ref device);
		if (commandQueue == IntPtr.Zero) {
			Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
			return;
		}

		// Create OpenCL program from kernel source
		program = CreateProgram(cl, context, device, @"D:\Programming\OpenTK\OpenCL Barnes-Hut\NBodyKernel.cl");
		if (program == IntPtr.Zero) {
			Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
			return;
		}

		// Create OpenCL kernel
		kernel = cl.CreateKernel(program, "nbody_step", null);
		if (kernel == IntPtr.Zero) {
			Console.WriteLine("Failed to create kernel");
			Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
			return;
		}

		// Create Memory Arrays
		var positions = new float[NumberOfBodies * 3];
		var velocities = new float[NumberOfBodies * 3];
		var masses = new float[NumberOfBodies];

		// Initialize data
		/*for (var i = 0; i < NumberOfBodies; i++) {
			positions[i] = i;
			velocities[i] = i * 2;
			masses[i] = i * 3;
		}*/

		/*masses[0] = 398576057600000; // Mass Earth
		masses[1] = 4904113984000; // Mass Moon


		positions[0] = 0; //Position Earth
		positions[1] = 0;
		positions[2] = 0;

		positions[3] = 384000000; //Position Moon
		positions[4] = 0;
		positions[5] = 0;


		velocities[0] = 0;
		velocities[1] = 0;
		velocities[2] = 0;

		velocities[3] = 0;
		velocities[4] = 1022;
		velocities[5] = 0;*/

		masses[0] = 4.297e6 * 2e30;
		positions[0] = 0; //Position Sagittarius A*
		positions[1] = 0;
		positions[2] = 0;
		velocities[0] = 0; //Velocity Sagittarius A*
		velocities[1] = 0;
		velocities[2] = 0;


		if (!CreateMemObjects(cl, context, memObjects, positions, velocities, masses)) {
			Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
			return;
		}


		nuint[] globalWorkSize = [NumberOfBodies];
		nuint[] localWorkSize = [1];

		var errNum = cl.SetKernelArg(kernel, 2, (nuint)sizeof(nint), memObjects[2]);

		Console.WriteLine($"{positions[0]}");

		writer.WriteLine($"{NumberOfBodies}");
		writer.WriteLine("0");
		for (var i = 0; i < NumberOfBodies; i++)
			writer.WriteLine($"{positions[i * 3]} {positions[i * 3 + 1]} {positions[i * 3 + 2]}");


		for (var i = 0; i < Iterations; i++) {
			// Set the kernel arguments (position, velocity, mass)
			errNum |= cl.SetKernelArg(kernel, 0, (nuint)sizeof(nint), memObjects[0]);
			errNum |= cl.SetKernelArg(kernel, 1, (nuint)sizeof(nint), memObjects[1]);

			if (errNum != (int)ErrorCodes.Success) {
				Console.WriteLine("Error setting kernel arguments.");
				Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
				return;
			}


			// Enqueue kernel for execution
			errNum = cl.EnqueueNdrangeKernel(commandQueue, kernel, 1, (nuint*)null, globalWorkSize, localWorkSize, 0,
				(nint*)null, (nint*)null);
			if (errNum != (int)ErrorCodes.Success) {
				Console.WriteLine("Error queuing kernel for execution.");
				Console.WriteLine(errNum.ToString());
				Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
				return;
			}


			// Extract data
			fixed (void* pPositions = positions) {
				errNum = cl.EnqueueReadBuffer(commandQueue, memObjects[0], true, 0, 3 * NumberOfBodies * sizeof(float),
					pPositions, 0, null, null);

				if (errNum != (int)ErrorCodes.Success) {
					Console.WriteLine("Error reading position buffer");
					Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
					return;
				}
			}

			fixed (void* pVelocities = velocities) {
				errNum = cl.EnqueueReadBuffer(commandQueue, memObjects[1], true, 0, 3 * NumberOfBodies * sizeof(float),
					pVelocities, 0, null, null);

				if (errNum != (int)ErrorCodes.Success) {
					Console.WriteLine("Error reading velocity buffer");
					Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
					return;
				}
			}

			writer.WriteLine($"{DeltaTime * (i + 1)}");
			for (var k = 0; k < NumberOfBodies; k++)
				writer.WriteLine($"{positions[k * 3]} {positions[k * 3 + 1]} {positions[k * 3 + 2]}");
		}


		Console.WriteLine("Executed program succesfully.");
		Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
	}


	// Create memory objects used as the arguments to the kernel
	private static unsafe bool CreateMemObjects(CL cl, nint context, nint[] memObjects, float[] positions,
		float[] velocities, float[] masses) {
		fixed (void* pPositions = positions) {
			memObjects[0] = cl.CreateBuffer(context, MemFlags.ReadWrite | MemFlags.CopyHostPtr,
				sizeof(float) * NumberOfBodies * 3, pPositions, null);
		}

		fixed (void* pVelocities = velocities) {
			memObjects[1] = cl.CreateBuffer(context, MemFlags.ReadWrite | MemFlags.CopyHostPtr,
				sizeof(float) * NumberOfBodies * 3, pVelocities, null);
		}

		fixed (void* pMasses = masses) {
			memObjects[2] = cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
				sizeof(float) * NumberOfBodies, pMasses, null);
		}

		if (memObjects[0] == IntPtr.Zero || memObjects[1] == IntPtr.Zero || memObjects[2] == IntPtr.Zero) {
			Console.WriteLine("Error creating memory objects.");
			return false;
		}

		return true;
	}


	// Create an OpenCL program from the kernel source file
	private static unsafe nint CreateProgram(CL cl, nint context, nint device, string fileName) {
		if (!File.Exists(fileName)) {
			Console.WriteLine($"File does not exist: {fileName}");
			return IntPtr.Zero;
		}

		using var sr = new StreamReader(fileName);
		var clStr = sr.ReadToEnd();

		var program = cl.CreateProgramWithSource(context, 1, new[] { clStr }, null, null);
		if (program == IntPtr.Zero) {
			Console.WriteLine("Failed to create CL program from source.");
			return IntPtr.Zero;
		}

		var errNum = cl.BuildProgram(program, 0, null, (byte*)null, null, null);

		if (errNum != (int)ErrorCodes.Success) {
			_ = cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, 0, null, out var buildLogSize);
			var log = new byte[buildLogSize / sizeof(byte)];
			fixed (void* pValue = log) {
				cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, buildLogSize, pValue, null);
			}

			var build_log = Encoding.UTF8.GetString(log);

			//Console.WriteLine("Error in kernel: ");
			Console.WriteLine("=============== OpenCL Program Build Info ================");
			Console.WriteLine(build_log);
			Console.WriteLine("==========================================================");

			cl.ReleaseProgram(program);
			return IntPtr.Zero;
		}

		return program;
	}

	// Cleanup resources
	private static void Cleanup(CL cl, nint context, nint commandQueue,
		nint program, nint kernel, nint[] memObjects, StreamWriter writer) {
		for (var i = 0; i < memObjects.Length; i++)
			if (memObjects[i] != 0)
				cl.ReleaseMemObject(memObjects[i]);
		if (commandQueue != 0)
			cl.ReleaseCommandQueue(commandQueue);

		if (kernel != 0)
			cl.ReleaseKernel(kernel);

		if (program != 0)
			cl.ReleaseProgram(program);

		if (context != 0)
			cl.ReleaseContext(context);

		writer.Dispose();
	}

	/// <summary>
	///     Create a command queue on the first device available on the
	///     context
	/// </summary>
	/// <param name="cL"></param>
	/// <param name="context"></param>
	/// <param name="device"></param>
	/// <returns></returns>
	private static unsafe nint CreateCommandQueue(CL cL, nint context, ref nint device) {
		var errNum = cL.GetContextInfo(context, ContextInfo.Devices, 0, null, out var deviceBufferSize);
		if (errNum != (int)ErrorCodes.Success) {
			Console.WriteLine("Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)");
			return IntPtr.Zero;
		}

		if (deviceBufferSize <= 0) {
			Console.WriteLine("No devices available.");
			return IntPtr.Zero;
		}

		var devices = new nint[deviceBufferSize / (nuint)sizeof(nuint)];
		fixed (void* pValue = devices) {
			var er = cL.GetContextInfo(context, ContextInfo.Devices, deviceBufferSize, pValue, null);

			if (er != (int)ErrorCodes.Success) {
				Console.WriteLine("Failed to get device IDs");
				return IntPtr.Zero;
			}
		}


		var commandQueue = cL.CreateCommandQueue(context, devices[0], CommandQueueProperties.None, null);
		if (commandQueue == IntPtr.Zero) {
			Console.WriteLine("Failed to create commandQueue for device 0");
			return IntPtr.Zero;
		}

		device = devices[0];
		return commandQueue;
	}

	/// <summary>
	///     Create an OpenCL context on the first available platform using
	///     either a GPU or CPU depending on what is available.
	/// </summary>
	/// <param name="cL"></param>
	/// <returns></returns>
	private static unsafe nint CreateContext(CL cL) {
		var errNum = cL.GetPlatformIDs(1, out var firstPlatformId, out var numPlatforms);
		if (errNum != (int)ErrorCodes.Success || numPlatforms <= 0) {
			Console.WriteLine("Failed to find any OpenCL platforms.");
			return IntPtr.Zero;
		}

		// Next, create an OpenCL context on the platform.  Attempt to
		// create a GPU-based context, and if that fails, try to create
		// a CPU-based context.
		nint[] contextProperties = {
			(nint)ContextProperties.Platform,
			firstPlatformId,
			0
		};

		fixed (nint* p = contextProperties) {
			var context = cL.CreateContextFromType(p, DeviceType.Gpu, null, null, out errNum);
			if (errNum != (int)ErrorCodes.Success) {
				Console.WriteLine("Could not create GPU context, trying CPU...");

				context = cL.CreateContextFromType(p, DeviceType.Cpu, null, null, out errNum);

				if (errNum != (int)ErrorCodes.Success) {
					Console.WriteLine("Failed to create an OpenCL GPU or CPU context.");
					return IntPtr.Zero;
				}

				return context;
			}

			return context;
		}
	}
}