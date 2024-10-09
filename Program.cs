using System.ComponentModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Text;
using NLog;
using Silk.NET.OpenCL;

namespace OpenCL_Barnes_Hut;

public enum IntegrationMethods {
	[Description("Euler Integration")] Euler,
	[Description("Verlet Integration")] Verlet,

	[Description("4th Order Runge-Kutta Integration")]
	RK4,

	[Description("5th Order P2 Method Multistep Integration")]
	M52
}

internal static class Program {
	private const int NumberOfBodies = 4;
	private const int Iterations = 250000;
	private const double TimeStep = 10;

	private const IntegrationMethods IntegrationMethod = IntegrationMethods.M52;
	private const UniverseSetups UniverseSetup = UniverseSetups.EarthMoonSatellites;

	private const int LogEvery = 1;
	private const int RepeatSim = 1;
	private const int ReferenceFrame = 0; // BodyID of reference frame

	private static readonly Logger Logger = LogManager.GetCurrentClassLogger();

	private static unsafe void Main() {
		var cl = CL.GetApi();
		nint nBodyKernel = 0;
		nint device = 0;
		nint shiftKernel = 0;

		var timeStep = TimeStep;
		var numberOfBodies = NumberOfBodies;

		LogManager.Setup().LoadConfiguration(builder => {
			builder.ForLogger().FilterMinLevel(LogLevel.Info).WriteToColoredConsole();
			builder.ForLogger().FilterMinLevel(LogLevel.Debug)
				.WriteToFile(@"D:\Programming\C#\OpenCL Barnes-Hut\Output\log.txt");
		});

		var csvWriter = new StreamWriter(@"D:\Programming\C#\OpenCL Barnes-Hut\Output\NBodyData.csv");
		var memObjects = new nint[3];

		// Create an OpenCL context
		var context = CreateContext(cl);
		if (context == IntPtr.Zero) {
			Logger.Fatal("Failed to create OpenCL context.");
			csvWriter.Dispose();
			return;
		}

		var commandQueue = CreateCommandQueue(cl, context, ref device);
		var program = CreateProgram(cl, context, device, @"D:\Programming\C#\OpenCL Barnes-Hut\Kernels\NBodyKernel.cl");
		if (program == IntPtr.Zero || commandQueue == IntPtr.Zero) {
			Cleanup(cl, context, commandQueue, program, nBodyKernel, memObjects, csvWriter, shiftKernel);
			return;
		}

		// ReSharper disable HeuristicUnreachableCode
		// Create OpenCL kernel
		nBodyKernel = IntegrationMethod switch {
			IntegrationMethods.Euler => cl.CreateKernel(program, "integrate_euler", null),
			IntegrationMethods.Verlet => cl.CreateKernel(program, "integrate_verlet", null),
			IntegrationMethods.RK4 => cl.CreateKernel(program, "integrate_rk4", null),
			IntegrationMethods.M52 => cl.CreateKernel(program, "integrate_multistep_5_2", null),
			_ => cl.CreateKernel(program, "integrate_euler", null)
		};
		shiftKernel = cl.CreateKernel(program, "shiftKernel", null);
		if (nBodyKernel == IntPtr.Zero || shiftKernel == IntPtr.Zero) {
			Logger.Fatal("Failed to create " + (nBodyKernel == IntPtr.Zero ? "nBodyKernel" : "shiftKernel"));
			Cleanup(cl, context, commandQueue, program, nBodyKernel, memObjects, csvWriter, shiftKernel);
			return;
		}
		// ReSharper restore HeuristicUnreachableCode

		switch (IntegrationMethod) {
			case IntegrationMethods.Euler:
				IntegrateNormal(cl, nBodyKernel, context, memObjects, commandQueue, program, timeStep, numberOfBodies,
					csvWriter);
				break;
			case IntegrationMethods.Verlet:
				IntegrateNormal(cl, nBodyKernel, context, memObjects, commandQueue, program, timeStep, numberOfBodies,
					csvWriter);
				break;
			case IntegrationMethods.RK4:
				IntegrateNormal(cl, nBodyKernel, context, memObjects, commandQueue, program, timeStep, numberOfBodies,
					csvWriter);
				break;
			case IntegrationMethods.M52:
				IntegrateMultistep(cl, nBodyKernel, context, memObjects, commandQueue, program, timeStep,
					numberOfBodies,
					csvWriter, shiftKernel);
				break;
		}

		Cleanup(cl, context, commandQueue, program, nBodyKernel, memObjects, csvWriter, shiftKernel);

		//OpenClWindow.RunWindow();
	}

	private static unsafe void IntegrateMultistep(CL cl, nint nBodyKernel, nint context, nint[] memObjects,
		nint commandQueue, nint program, double deltaTime, int numberOfBodies, StreamWriter writer, nint shiftKernel) {
		var (positions, _, masses) =
			Universe.GetUniverse(NumberOfBodies, UniverseSetup, TimeStep);

		// Turn data into memory objects
		if (!CreateMemObjectsMultistep(cl, context, memObjects, positions, masses)) {
			Cleanup(cl, context, commandQueue, program, nBodyKernel, memObjects, writer, shiftKernel);
			return;
		}

		// Set the nBodyKernel arguments (positions, velocities, masses, timestep, body count)
		var errNum = cl.SetKernelArg(nBodyKernel, 0, (nuint)sizeof(nint), memObjects[0]);
		errNum |= cl.SetKernelArg(nBodyKernel, 1, (nuint)sizeof(nint), memObjects[1]);
		errNum |= cl.SetKernelArg(nBodyKernel, 2, sizeof(double), &deltaTime);
		errNum |= cl.SetKernelArg(nBodyKernel, 3, sizeof(int), &numberOfBodies);

		// Set the ShiftKernel arguments (positions, numberOfBodies)
		var errNumS = cl.SetKernelArg(shiftKernel, 0, (nuint)sizeof(nint), memObjects[0]);
		errNumS |= cl.SetKernelArg(shiftKernel, 1, sizeof(int), &numberOfBodies);

		if (errNumS != (int)ErrorCodes.Success) {
			Logger.Fatal("Error setting shiftKernel arguments.");
			Cleanup(cl, context, commandQueue, program, nBodyKernel, memObjects, writer, shiftKernel);
			return;
		}

		if (errNum != (int)ErrorCodes.Success) {
			Logger.Fatal("Error setting kernel arguments.");
			Cleanup(cl, context, commandQueue, program, nBodyKernel, memObjects, writer, shiftKernel);
			return;
		}

		// Write start data to csv
		writer.WriteLine("time,bodyId,xPosition,yPosition,zPosition");
		SavePositionData(writer, positions, 0, NumberOfBodies, 1);

		Logger.Info("Starting N-Body simulation.");
		var stopwatch = Stopwatch.StartNew();

		for (var repeat = 0; repeat < RepeatSim; repeat++)
		for (var iteration = 0; iteration < Iterations; iteration++) {
			// Enqueue kernels for execution
			cl.EnqueueNdrangeKernel(commandQueue, nBodyKernel, 1, (nuint*)null, [NumberOfBodies], [1], 0,
				(nint*)null, (nint*)null);

			cl.EnqueueNdrangeKernel(commandQueue, shiftKernel, 1, (nuint*)null, [1], [1], 0,
				(nint*)null, (nint*)null);

			if (iteration % LogEvery == 0) {
				fixed (void* pPositions = positions) {
					errNum = cl.EnqueueReadBuffer(commandQueue, memObjects[0], true, 0,
						NumberOfBodies * sizeof(double) * 4 * 5,
						pPositions, 0, null, null);
				}

				if (errNum != (int)ErrorCodes.Success) {
					Logger.Fatal("Error reading position or velocity buffer.");
					Cleanup(cl, context, commandQueue, program, nBodyKernel, memObjects, writer, shiftKernel);
					return;
				}

				SavePositionData(writer, positions, iteration, NumberOfBodies, 1);
			}
		}

		stopwatch.Stop();

		Logger.Info(
			$" Executed program succesfully, data:\n" +
			$"                                                       |     - Time elapsed: {stopwatch.Elapsed.TotalSeconds} seconds\n" +
			$"                                                       |     - Iterations: {Iterations}\n" +
			$"                                                       |     - Integration Method: {GetEnumDescription(IntegrationMethod)}\n" +
			$"                                                       |     - Number of bodies interacting: {NumberOfBodies}\n" +
			$"                                                       |     - Logged to CSV every {LogEvery} integration cycles\n" +
			$"                                                       |     - Repeated full simulation {RepeatSim} times\n" +
			$"                                                       |     - Average simulation runtime: {stopwatch.Elapsed.TotalSeconds / RepeatSim} seconds\n" +
			$"                                                       |     - Average iteration runtime: {stopwatch.Elapsed.TotalSeconds / (RepeatSim * Iterations)} seconds\n"
		);
	}

	private static unsafe void IntegrateNormal(CL cl, nint nBodyKernel, nint context, nint[] memObjects,
		nint commandQueue, nint program, double deltaTime, int numberOfBodies, StreamWriter csvWriter) {
		var (positions, velocities, masses) =
			Universe.GetUniverse(NumberOfBodies, UniverseSetup);

		// Turn data into memory objects
		if (!CreateMemObjectsNormal(cl, context, memObjects, positions, velocities, masses)) {
			Cleanup(cl, context, commandQueue, program, nBodyKernel, memObjects, csvWriter, 0);
			return;
		}

		// Set the kernel arguments (position, velocity, mass, dt, body count)
		var errNum = cl.SetKernelArg(nBodyKernel, 0, (nuint)sizeof(nint), memObjects[0]);
		errNum |= cl.SetKernelArg(nBodyKernel, 1, (nuint)sizeof(nint), memObjects[1]);
		errNum |= cl.SetKernelArg(nBodyKernel, 2, (nuint)sizeof(nint), memObjects[2]);
		errNum |= cl.SetKernelArg(nBodyKernel, 3, sizeof(double), &deltaTime);
		errNum |= cl.SetKernelArg(nBodyKernel, 4, sizeof(int), &numberOfBodies);

		if (errNum != (int)ErrorCodes.Success) {
			Logger.Fatal("Error setting kernel arguments.");
			Cleanup(cl, context, commandQueue, program, nBodyKernel, memObjects, csvWriter, 0);
			return;
		}

		// Write start data to csv
		csvWriter.WriteLine("time,bodyId,xPosition,yPosition,zPosition");
		SavePositionData(csvWriter, positions, 0, NumberOfBodies, 0);

		nuint[] globalWorkSize = [NumberOfBodies];
		nuint[] localWorkSize = [1];

		Logger.Info("Starting N-Body simulation.");
		var stopwatch = Stopwatch.StartNew();

		for (var repeat = 0; repeat < RepeatSim; repeat++)
		for (var iteration = 0; iteration < Iterations; iteration++) {
			// Enqueue kernel for execution
			cl.EnqueueNdrangeKernel(commandQueue, nBodyKernel, 1, (nuint*)null, globalWorkSize, localWorkSize, 0,
				(nint*)null, (nint*)null);

			if (iteration % LogEvery == 0) {
				fixed (void* pPositions = positions) {
					errNum = cl.EnqueueReadBuffer(commandQueue, memObjects[0], true, 0,
						NumberOfBodies * sizeof(double) * 4,
						pPositions, 0, null, null);
				}

				if (errNum != (int)ErrorCodes.Success) {
					Logger.Fatal("Error reading position or velocity buffer.");
					Cleanup(cl, context, commandQueue, program, nBodyKernel, memObjects, csvWriter, 0);
					return;
				}

				SavePositionData(csvWriter, positions, iteration, NumberOfBodies, 0);
			}
		}

		stopwatch.Stop();

		Logger.Info(
			$" Executed program succesfully, data:\n" +
			$"                                                       |     - Time elapsed: {stopwatch.Elapsed.TotalSeconds} seconds\n" +
			$"                                                       |     - Iterations: {Iterations}\n" +
			$"                                                       |     - Integration Method: {GetEnumDescription(IntegrationMethod)}\n" +
			$"                                                       |     - Number of bodies interacting: {NumberOfBodies}\n" +
			$"                                                       |     - Logged to CSV every {LogEvery} integration cycles\n" +
			$"                                                       |     - Repeated full simulation {RepeatSim} times\n" +
			$"                                                       |     - Average simulation runtime: {stopwatch.Elapsed.TotalSeconds / RepeatSim} seconds\n" +
			$"                                                       |     - Average iteration runtime: {stopwatch.Elapsed.TotalSeconds / (RepeatSim * Iterations)} seconds\n"
		);
	}

	private static void SavePositionData(StreamWriter csvWriter, double[] positions, int iteration, int numberOfBodies,
		int multistep) {
		// ReSharper disable RedundantAssignment
		var posx = 0.0;
		var posy = 0.0;
		var posz = 0.0;
		// ReSharper restore RedundantAssignment

		if (ReferenceFrame != -1) {
			posx = positions[multistep * numberOfBodies * 4 + ReferenceFrame * 4];
			posy = positions[multistep * numberOfBodies * 4 + ReferenceFrame * 4 + 1];
			posz = positions[multistep * numberOfBodies * 4 + ReferenceFrame * 4 + 2];
		}

		for (var i = 0; i < NumberOfBodies; i++)
			csvWriter.WriteLine($"{DoubleToString((iteration + 1) * TimeStep, "g")}," +
			                    $"{DoubleToString(i, "g")}," +
			                    $"{DoubleToString(
				                    positions[multistep * numberOfBodies * 4 + i * 4 + 0] - posx,
				                    "E")}," +
			                    $"{DoubleToString(
				                    positions[multistep * numberOfBodies * 4 + i * 4 + 1] - posy,
				                    "E")}," +
			                    $"{DoubleToString(
				                    positions[multistep * numberOfBodies * 4 + i * 4 + 2] - posz,
				                    "E")}");
	}


	// Create memory objects used as the arguments to the kernel
	private static unsafe bool CreateMemObjectsNormal(CL cl, nint context, nint[] memObjects,
		double[] positions, double[] velocities, double[] masses
	) {
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

	private static unsafe bool CreateMemObjectsMultistep(CL cl, nint context, nint[] memObjects,
		double[] positions, double[] masses
	) {
		fixed (void* pPositions = positions) {
			memObjects[0] = cl.CreateBuffer(context, MemFlags.ReadWrite | MemFlags.CopyHostPtr,
				NumberOfBodies * sizeof(double) * 4 * 5, pPositions, null);
		}

		fixed (void* pMasses = masses) {
			memObjects[1] = cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.HostWriteOnly | MemFlags.CopyHostPtr,
				sizeof(double) * NumberOfBodies, pMasses, null);
		}

		if (memObjects[0] != IntPtr.Zero && memObjects[1] != IntPtr.Zero) return true;

		Logger.Fatal("Error creating memory objects.");
		return false;
	}


	// Create an OpenCL program from the kernel source file
	private static unsafe nint CreateProgram(CL cl, nint context, nint device, string kernelFile) {
		if (!File.Exists(kernelFile)) {
			Logger.Fatal($"File does not exist: {kernelFile}");
			return IntPtr.Zero;
		}

		using var sr = new StreamReader(kernelFile);
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
		nint program, nint nBodyKernel, nint[] memObjects, StreamWriter csvWriter, nint shiftKernel) {
		foreach (var memObject in memObjects)
			if (memObject != 0)
				cl.ReleaseMemObject(memObject);

		if (commandQueue != 0)
			cl.ReleaseCommandQueue(commandQueue);

		if (nBodyKernel != 0)
			cl.ReleaseKernel(nBodyKernel);

		if (program != 0)
			cl.ReleaseProgram(program);

		if (context != 0)
			cl.ReleaseContext(context);

		if (shiftKernel != 0)
			cl.ReleaseKernel(shiftKernel);

		csvWriter.Dispose();
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

	private static string DoubleToString(double number, [StringSyntax("NumericFormat")] string numericFormat) {
		return number.ToString(numericFormat, CultureInfo.InvariantCulture);
	}

	private static string GetEnumDescription(Enum e) {
		var fieldInfo = e.GetType().GetField(e.ToString());
		return fieldInfo?.GetCustomAttributes(typeof(DescriptionAttribute), false) is DescriptionAttribute[]
			enumAttributes
			? enumAttributes[0].Description
			: e.ToString();
	}
}