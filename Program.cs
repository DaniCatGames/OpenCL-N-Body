using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Text;
using NLog;
using Silk.NET.OpenCL;

namespace OpenCL_Barnes_Hut;

internal struct double3 {
	public double X;
	public double Y;
	public double Z;

	public double3(double x) {
		X = x;
		Y = x;
		Z = x;
	}

	public double3(double x, double y, double z) {
		X = x;
		Y = y;
		Z = z;
	}
}

internal class Program {
	private const int NumberOfBodies = 2;
	private const int Iterations = 2;
	private const double DeltaTime = 0.001;
	private const int logEvery = 1;

	private const double G = 6.674315e-11;
	private static readonly Logger logger = LogManager.GetCurrentClassLogger();

	private static unsafe void Main(string[] args) {
		var cl = CL.GetApi();
		nint context = 0;
		nint commandQueue = 0;
		nint program = 0;
		nint kernel = 0;
		nint device = 0;

		LogManager.Setup().LoadConfiguration(builder => {
			builder.ForLogger().FilterMinLevel(LogLevel.Info).WriteToConsole();
			builder.ForLogger().FilterMinLevel(LogLevel.Debug)
				.WriteToFile(@"D:\Programming\C#\OpenCL Barnes-Hut\Output\log.txt");
		});

		logger.Debug("Starting StreamWriter");
		var nBodyDataCsv = @"D:\Programming\C#\OpenCL Barnes-Hut\Output\NBodyData.csv";
		var writer = new StreamWriter(nBodyDataCsv);

		var memObjects = new nint[3];

		// Create an OpenCL context on first available platform
		context = CreateContext(cl);
		if (context == IntPtr.Zero) {
			logger.Fatal("Failed to create OpenCL context.");
			writer.Dispose();
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
		program = CreateProgram(cl, context, device, @"D:\Programming\C#\OpenCL Barnes-Hut\Kernels\NBodyKernel.cl");
		if (program == IntPtr.Zero) {
			Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
			return;
		}

		// Create OpenCL kernel
		kernel = cl.CreateKernel(program, "nbody_step", null);
		if (kernel == IntPtr.Zero) {
			logger.Fatal("Failed to create kernel.");
			Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
			return;
		}

		// Create Memory Arrays
		var positions = new double3[NumberOfBodies];
		var velocities = new double3[NumberOfBodies];
		var masses = new double[NumberOfBodies];

		// Initialize data
		masses[0] = 5.972e24 * G; // Mass Earth
		masses[1] = 7.348e22 * G; // Mass Moon 

		positions[0] = new double3(0.0); //Position Earth
		positions[1] = new double3(3.84e8, 0.0, 0.0); //Position Moon

		velocities[0] = new double3(0.0);
		velocities[1] = new double3(0.0, 1022.0, 0.0);

		/*
		masses[0] = 4.297e6 * 2e30 * 6.67408e-11;
		positions[0] = new double3(0.0); //Position Sagittarius A*
		velocities[0] = new double3(0.0); //Velocity Sagittarius A*

		for (var i = 1; i < NumberOfBodies; i++) {
			masses[i] = 2e30 * G;
			positions[i] = new double3(100e9 + i * 1e9, 0.0, 0.0);
			velocities[i] = new double3(0.0, double.Sqrt(G * masses[0] / positions[i].X), 0.0);
		}*/


		var flattenedPositions = FlattenDoubleArray(positions);
		var flattenedVelocities = FlattenDoubleArray(velocities);

		if (!CreateMemObjects(cl, context, memObjects, flattenedPositions, flattenedVelocities, masses)) {
			Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
			return;
		}


		nuint[] globalWorkSize = [NumberOfBodies];
		nuint[] localWorkSize = [1];

		// Set the kernel arguments (mass, dt, body count)
		var errNum = cl.SetKernelArg(kernel, 2, (nuint)sizeof(nint), memObjects[2]);
		//errNum |= cl.SetKernelArg(kernel, 3, sizeof(float), DeltaTime);
		//errNum |= cl.SetKernelArg(kernel, 4, sizeof(int), NumberOfBodies);

		//Write start data to csv
		writer.WriteLine("bodyId,time,xPosition,yPosition,zPosition");
		for (var k = 0; k < NumberOfBodies; k++)
			writer.WriteLine(DoubleToString(k, "g") + ",0," + DoubleToString(flattenedPositions[k * 3], "e2") + "," +
			                 DoubleToString(flattenedPositions[k * 3 + 1], "e2") + "," +
			                 DoubleToString(flattenedPositions[k * 3 + 2], "e2"));


		logger.Info("Starting N-Body simulation.");
		var stopwatch = Stopwatch.StartNew();

		for (var i = 0; i < Iterations; i++) {
			// Set the kernel arguments (position, velocity)
			errNum |= cl.SetKernelArg(kernel, 0, (nuint)sizeof(nint), memObjects[0]);
			errNum |= cl.SetKernelArg(kernel, 1, (nuint)sizeof(nint), memObjects[1]);

			if (errNum != (int)ErrorCodes.Success) {
				logger.Fatal("Error setting kernel arguments.");
				Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
				return;
			}


			// Enqueue kernel for execution
			errNum = cl.EnqueueNdrangeKernel(commandQueue, kernel, 1, (nuint*)null, globalWorkSize, localWorkSize, 0,
				(nint*)null, (nint*)null);
			if (errNum != (int)ErrorCodes.Success) {
				logger.Fatal("Error queuing kernel for execution.");
				logger.Debug(errNum.ToString());
				Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
				return;
			}

			// Extract data
			fixed (void* pPositions = flattenedPositions) {
				errNum = cl.EnqueueReadBuffer(commandQueue, memObjects[0], true, 0,
					NumberOfBodies * sizeof(double) * 3,
					pPositions, 0, null, null);

				if (errNum != (int)ErrorCodes.Success) {
					logger.Fatal("Error reading position buffer.");
					Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
					return;
				}
			}

			fixed (void* pVelocities = flattenedVelocities) {
				errNum = cl.EnqueueReadBuffer(commandQueue, memObjects[1], true, 0,
					NumberOfBodies * sizeof(double) * 3,
					pVelocities, 0, null, null);

				if (errNum != (int)ErrorCodes.Success) {
					logger.Fatal("Error reading velocity buffer.");
					Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
					return;
				}
			}

			for (var k = 0; k < NumberOfBodies; k++)
				logger.Info(
					$"Iteration {i}, Body {k}: Position ({flattenedPositions[k * 3]}, {flattenedPositions[k * 3 + 1]}, {flattenedPositions[k * 3 + 2]}), \n" +
					$"Velocity ({flattenedVelocities[k * 3]}, {flattenedVelocities[k * 3 + 1]}, {flattenedVelocities[k * 3 + 2]})");

			//Write step data to csv file
			for (var k = 0; k < NumberOfBodies; k++)
				if (i % logEvery == 0)
					writer.WriteLine(DoubleToString(k, "g") + "," + DoubleToString((i + 1) * DeltaTime, "g") + "," +
					                 DoubleToString(flattenedPositions[k * 3], "e2") + "," +
					                 DoubleToString(flattenedPositions[k * 3 + 1], "e2") + "," +
					                 DoubleToString(flattenedPositions[k * 3 + 2], "e2"));

			logger.Debug($"Stepped N-Body Sim, iteration: {i + 1}");
		}

		stopwatch.Stop();

		logger.Info(
			$"Executed program succesfully, time elapsed: {stopwatch.ElapsedMilliseconds / 1000.0f} seconds");
		Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);

		//OpenClWindow.RunWindow();
	}


	// Create memory objects used as the arguments to the kernel
	private static unsafe bool CreateMemObjects(CL cl, nint context, nint[] memObjects,
		double[] positions, double[] velocities, double[] masses) {
		fixed (void* pPositions = positions) {
			memObjects[0] = cl.CreateBuffer(context, MemFlags.ReadWrite | MemFlags.CopyHostPtr,
				NumberOfBodies * sizeof(double) * 3, pPositions, null);
		}

		fixed (void* pVelocities = velocities) {
			memObjects[1] = cl.CreateBuffer(context, MemFlags.ReadWrite | MemFlags.CopyHostPtr,
				NumberOfBodies * sizeof(double) * 3, pVelocities, null);
		}

		fixed (void* pMasses = masses) {
			memObjects[2] = cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.HostWriteOnly | MemFlags.CopyHostPtr,
				sizeof(double) * NumberOfBodies, pMasses, null);
		}

		if (memObjects[0] == IntPtr.Zero || memObjects[1] == IntPtr.Zero || memObjects[2] == IntPtr.Zero) {
			logger.Fatal("Error creating memory objects.");
			return false;
		}

		return true;
	}


	// Create an OpenCL program from the kernel source file
	private static unsafe nint CreateProgram(CL cl, nint context, nint device, string fileName) {
		if (!File.Exists(fileName)) {
			logger.Fatal($"File does not exist: {fileName}");
			return IntPtr.Zero;
		}

		using var sr = new StreamReader(fileName);
		var clStr = sr.ReadToEnd();

		var program = cl.CreateProgramWithSource(context, 1, new[] { clStr }, null, null);
		if (program == IntPtr.Zero) {
			logger.Fatal("Failed to create CL program from source.");
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

			logger.Fatal("Error in kernel.");
			logger.Info("=============== OpenCL Program Build Info ================");
			logger.Info(build_log);
			logger.Info("==========================================================");

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


	//Create a command queue on the first device available on the context
	private static unsafe nint CreateCommandQueue(CL cL, nint context, ref nint device) {
		var errNum = cL.GetContextInfo(context, ContextInfo.Devices, 0, null, out var deviceBufferSize);
		if (errNum != (int)ErrorCodes.Success) {
			logger.Fatal("Failed call to clGetContextInfo(...,GL_CONTEXT_DEVICES,...)");
			return IntPtr.Zero;
		}

		if (deviceBufferSize <= 0) {
			logger.Fatal("No devices available.");
			return IntPtr.Zero;
		}

		var devices = new nint[deviceBufferSize / (nuint)sizeof(nuint)];
		fixed (void* pValue = devices) {
			var er = cL.GetContextInfo(context, ContextInfo.Devices, deviceBufferSize, pValue, null);

			if (er != (int)ErrorCodes.Success) {
				logger.Fatal("Failed to get device IDs");
				return IntPtr.Zero;
			}
		}


		var commandQueue = cL.CreateCommandQueue(context, devices[0], CommandQueueProperties.None, null);
		if (commandQueue == IntPtr.Zero) {
			logger.Fatal("Failed to create commandQueue for device 0");
			return IntPtr.Zero;
		}

		device = devices[0];
		return commandQueue;
	}

	// Create an OpenCL context on the first available platform using either a GPU or CPU depending on what is available.
	private static unsafe nint CreateContext(CL cL) {
		var errNum = cL.GetPlatformIDs(1, out var firstPlatformId, out var numPlatforms);
		if (errNum != (int)ErrorCodes.Success || numPlatforms <= 0) {
			logger.Fatal("Failed to find any OpenCL platforms.");
			return IntPtr.Zero;
		}

		// Next, create an OpenCL context on the platform.  Attempt to create a GPU-based context, and if that fails, try to create a CPU-based context.
		nint[] contextProperties = {
			(nint)ContextProperties.Platform,
			firstPlatformId,
			0
		};

		fixed (nint* p = contextProperties) {
			var context = cL.CreateContextFromType(p, DeviceType.Gpu, null, null, out errNum);
			if (errNum != (int)ErrorCodes.Success) {
				logger.Info("Could not create GPU context, trying to create CPU context.");

				context = cL.CreateContextFromType(p, DeviceType.Cpu, null, null, out errNum);

				if (errNum != (int)ErrorCodes.Success) {
					logger.Fatal("Failed to create an OpenCL GPU or CPU context.");
					return IntPtr.Zero;
				}

				return context;
			}

			return context;
		}
	}

	private static string DoubleToString(double number, [StringSyntax("NumericFormat")] string format) {
		return number.ToString(format, CultureInfo.InvariantCulture);
	}

	private static double[] FlattenDoubleArray(double3[] array) {
		var flattened = new double[array.Length * 3];
		for (var i = 0; i < array.Length; i++) {
			flattened[i * 3] = array[i].X;
			flattened[i * 3 + 1] = array[i].Y;
			flattened[i * 3 + 2] = array[i].Z;
		}

		return flattened;
	}
}