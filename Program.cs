using System.ComponentModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Text;
using NLog;
using Silk.NET.OpenCL;

namespace OpenCL_Barnes_Hut;

internal enum IntegrationMethod {
	[Description("Euler Integration")] Euler,
	[Description("Verlet Integration")] Verlet,

	[Description("4th Order Runge-Kutta Integration")]
	RK4
}

internal static class Program {
	private const int NumberOfBodies = 100;
	private const int Iterations = 3000;
	private const double DeltaTime = 1;
	private const int LocalSize = 1;

	private const IntegrationMethod IntegrationMethodConfig = IntegrationMethod.Euler;
	private const UniverseSetup UniverseSetupConfig = UniverseSetup.EarthMoonSatellites;

	private const int LogEvery = 1;
	private const int RepeatSim = 5;
	private const int ReferenceFrame = 0; // BodyID of reference frame

	private static readonly Logger Logger = LogManager.GetCurrentClassLogger();

	private static unsafe void Main() {
		var cl = CL.GetApi();
		nint kernel = 0;
		nint device = 0;

		var deltaTime = DeltaTime;
		var numberOfBodies = NumberOfBodies;

		LogManager.Setup().LoadConfiguration(builder => {
			builder.ForLogger().FilterMinLevel(LogLevel.Info).WriteToColoredConsole();
			builder.ForLogger().FilterMinLevel(LogLevel.Debug)
				.WriteToFile(@"D:\Programming\C#\OpenCL Barnes-Hut\Output\log.txt");
		});

		var writer = new StreamWriter(@"D:\Programming\C#\OpenCL Barnes-Hut\Output\NBodyData.csv");
		var memObjects = new nint[3];

		// Create an OpenCL context
		var context = CreateContext(cl);
		if (context == IntPtr.Zero) {
			Logger.Fatal("Failed to create OpenCL context.");
			writer.Dispose();
			return;
		}

		var commandQueue = CreateCommandQueue(cl, context, ref device);
		var program = CreateProgram(cl, context, device, @"D:\Programming\C#\OpenCL Barnes-Hut\Kernels\NBodyKernel.cl");
		if (program == IntPtr.Zero || commandQueue == IntPtr.Zero) {
			Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
			return;
		}

		// ReSharper disable HeuristicUnreachableCode
		// Create OpenCL kernel
		kernel = IntegrationMethodConfig switch {
			IntegrationMethod.Euler => cl.CreateKernel(program, "integrate_euler", null),
			IntegrationMethod.Verlet => cl.CreateKernel(program, "integrate_verlet", null),
			IntegrationMethod.RK4 => cl.CreateKernel(program, "integrate_rk4", null),
			_ => cl.CreateKernel(program, "integrate_euler", null)
		};
		if (kernel == IntPtr.Zero) {
			Logger.Fatal("Failed to create kernel.");
			Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
			return;
		}
		// ReSharper restore HeuristicUnreachableCode

		var (positions, velocities, masses) = Universe.GetUniverse(NumberOfBodies, UniverseSetupConfig);

		// Turn data into memory objects
		if (!CreateMemObjects(cl, context, memObjects, positions, velocities, masses)) {
			Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
			return;
		}

		// Set the kernel arguments (position, velocity, mass, dt, body count)
		var errNum = cl.SetKernelArg(kernel, 0, (nuint)sizeof(nint), memObjects[0]);
		errNum |= cl.SetKernelArg(kernel, 1, (nuint)sizeof(nint), memObjects[1]);
		errNum |= cl.SetKernelArg(kernel, 2, (nuint)sizeof(nint), memObjects[2]);
		errNum |= cl.SetKernelArg(kernel, 3, sizeof(double), &deltaTime);
		errNum |= cl.SetKernelArg(kernel, 4, sizeof(int), &numberOfBodies);
		errNum |= cl.SetKernelArg(kernel, 5, LocalSize * 4 * sizeof(double), null);
		errNum |= cl.SetKernelArg(kernel, 6, LocalSize * sizeof(double), null);

		if (errNum != (int)ErrorCodes.Success) {
			Logger.Fatal("Error setting kernel arguments.");
			Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
			return;
		}

		// Write start data to csv
		writer.WriteLine("time,bodyId,xPosition,yPosition,zPosition");
		SavePositionData(writer, positions, 0);

		nuint[] globalWorkSize = [NumberOfBodies];
		nuint[] localWorkSize = [1];

		Logger.Info("Starting N-Body simulation.");
		var stopwatch = Stopwatch.StartNew();

		for (var repeat = 0; repeat < RepeatSim; repeat++)
		for (var iteration = 0; iteration < Iterations; iteration++) {
			// Enqueue kernel for execution
			cl.EnqueueNdrangeKernel(commandQueue, kernel, 1, (nuint*)null, globalWorkSize, localWorkSize, 0,
				(nint*)null, (nint*)null);

			if (iteration % LogEvery == 0) {
				fixed (void* pPositions = positions) {
					errNum = cl.EnqueueReadBuffer(commandQueue, memObjects[0], true, 0,
						NumberOfBodies * sizeof(double) * 4,
						pPositions, 0, null, null);
				}

				fixed (void* pVelocities = velocities) {
					errNum |= cl.EnqueueReadBuffer(commandQueue, memObjects[1], true, 0,
						NumberOfBodies * sizeof(double) * 4,
						pVelocities, 0, null, null);
				}

				if (errNum != (int)ErrorCodes.Success) {
					Logger.Fatal("Error reading position or velocity buffer.");
					Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);
					return;
				}

				SavePositionData(writer, positions, iteration);
			}
		}

		stopwatch.Stop();

		Logger.Info(
			$" Executed program succesfully, data:\n" +
			$"                                                       |     - Time elapsed: {stopwatch.Elapsed.TotalSeconds} seconds\n" +
			$"                                                       |     - Iterations: {Iterations}\n" +
			$"                                                       |     - Integration Method: {GetEnumDescription(IntegrationMethodConfig)}\n" +
			$"                                                       |     - Number of bodies interacting: {NumberOfBodies}\n" +
			$"                                                       |     - Logged to CSV every {LogEvery} integration cycles\n" +
			$"                                                       |     - Repeated full simulation {RepeatSim} times\n" + 
			$"                                                       |     - Average simulation runtime: {stopwatch.Elapsed.TotalSeconds / RepeatSim} seconds\n" + 
			$"                                                       |     - Average iteration runtime: {stopwatch.Elapsed.TotalSeconds / (RepeatSim * Iterations)} seconds\n"
		);
		Cleanup(cl, context, commandQueue, program, kernel, memObjects, writer);

		//OpenClWindow.RunWindow();
	}

	private static void SavePositionData(StreamWriter writer, double[] positions, int iteration) {
		for (var i = 0; i < NumberOfBodies; i++)
			writer.WriteLine(new StringBuilder().Append(DoubleToString((iteration + 1) * DeltaTime, "g"))
				.Append(',')
				.Append(DoubleToString(i, "g"))
				.Append(',')
				.Append(DoubleToString(
					positions[i * 4 + 0] - positions[ReferenceFrame * 4 + 0],
					"e2"))
				.Append(',')
				.Append(DoubleToString(
					positions[i * 4 + 1] - positions[ReferenceFrame * 4 + 1],
					"e2"))
				.Append(',')
				.Append(DoubleToString(
					positions[i * 4 + 2] - positions[ReferenceFrame * 4 + 2],
					"e2"))
				.ToString());
	}


	// Create memory objects used as the arguments to the kernel
	private static unsafe bool CreateMemObjects(CL cl, nint context, nint[] memObjects,
		double[] positions, double[] velocities, double[] masses) {
		fixed (void* pPositions = positions) {
			memObjects[0] = cl.CreateBuffer(context, MemFlags.ReadWrite | MemFlags.CopyHostPtr,
				NumberOfBodies * sizeof(double) * 4, pPositions, null);
		}

		fixed (void* pVelocities = velocities) {
			memObjects[1] = cl.CreateBuffer(context, MemFlags.ReadWrite | MemFlags.CopyHostPtr,
				NumberOfBodies * sizeof(double) * 4, pVelocities, null);
		}

		fixed (void* pMasses = masses) {
			memObjects[2] = cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.HostWriteOnly | MemFlags.CopyHostPtr,
				sizeof(double) * NumberOfBodies, pMasses, null);
		}

		if (memObjects[0] != IntPtr.Zero && memObjects[1] != IntPtr.Zero && memObjects[2] != IntPtr.Zero) return true;

		Logger.Fatal("Error creating memory objects.");
		return false;
	}


	// Create an OpenCL program from the kernel source file
	private static unsafe nint CreateProgram(CL cl, nint context, nint device, string fileName) {
		if (!File.Exists(fileName)) {
			Logger.Fatal($"File does not exist: {fileName}");
			return IntPtr.Zero;
		}

		using var sr = new StreamReader(fileName);
		var clStr = sr.ReadToEnd();

		var program = cl.CreateProgramWithSource(context, 1, [clStr], null, null);
		if (program == IntPtr.Zero) {
			Logger.Fatal("Failed to create CL program from source.");
			return IntPtr.Zero;
		}

		var errNum = cl.BuildProgram(program, 0, null, (byte*)null, null, null);

		if (errNum == (int)ErrorCodes.Success) return program;

		cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, 0, null, out var buildLogSize);
		var log = new byte[buildLogSize / sizeof(byte)];
		fixed (void* pValue = log) {
			cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.BuildLog, buildLogSize, pValue, null);
		}

		Logger.Fatal("Error in kernel.");
		Logger.Info(Encoding.UTF8.GetString(log));

		cl.ReleaseProgram(program);
		return IntPtr.Zero;
	}

	// Cleanup resources
	private static void Cleanup(CL cl, nint context, nint commandQueue,
		nint program, nint kernel, nint[] memObjects, StreamWriter writer) {
		foreach (var memObject in memObjects)
			if (memObject != 0)
				cl.ReleaseMemObject(memObject);

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
			Logger.Fatal("Failed call to clGetContextInfo");
			return IntPtr.Zero;
		}

		if (deviceBufferSize <= 0) {
			Logger.Fatal("No devices available.");
			return IntPtr.Zero;
		}

		var devices = new nint[deviceBufferSize / (nuint)sizeof(nuint)];
		fixed (void* pValue = devices) {
			var er = cL.GetContextInfo(context, ContextInfo.Devices, deviceBufferSize, pValue, null);

			if (er != (int)ErrorCodes.Success) {
				Logger.Fatal("Failed to get device IDs");
				return IntPtr.Zero;
			}
		}


		var commandQueue = cL.CreateCommandQueue(context, devices[0], CommandQueueProperties.None, null);
		if (commandQueue == IntPtr.Zero) {
			Logger.Fatal("Failed to create commandQueue for device 0");
			return IntPtr.Zero;
		}

		device = devices[0];
		return commandQueue;
	}

	// Create an OpenCL context on the first available platform using either a GPU or CPU depending on what is available.
	private static unsafe nint CreateContext(CL cL) {
		var errNum = cL.GetPlatformIDs(1, out var firstPlatformId, out var numPlatforms);
		if (errNum != (int)ErrorCodes.Success || numPlatforms <= 0) {
			Logger.Fatal("Failed to find any OpenCL platforms.");
			return IntPtr.Zero;
		}

		// Next, create an OpenCL context on the platform.  Attempt to create a GPU-based context, and if that fails, try to create a CPU-based context.
		nint[] contextProperties = [
			(nint)ContextProperties.Platform,
			firstPlatformId,
			0
		];

		fixed (nint* p = contextProperties) {
			var context = cL.CreateContextFromType(p, DeviceType.Gpu, null, null, out errNum);
			if (errNum == (int)ErrorCodes.Success) return context;

			Logger.Info("Could not create GPU context, trying to create CPU context.");

			context = cL.CreateContextFromType(p, DeviceType.Cpu, null, null, out errNum);

			if (errNum == (int)ErrorCodes.Success) return context;

			Logger.Fatal("Failed to create an OpenCL GPU or CPU context.");
			return IntPtr.Zero;
		}
	}

	private static string DoubleToString(double number, [StringSyntax("NumericFormat")] string format) {
		return number.ToString(format, CultureInfo.InvariantCulture);
	}

	private static string GetEnumDescription(Enum e) {
		var fieldInfo = e.GetType().GetField(e.ToString());
		return fieldInfo?.GetCustomAttributes(typeof(DescriptionAttribute), false) is DescriptionAttribute[]
			enumAttributes
			? enumAttributes[0].Description
			: e.ToString();
	}
}