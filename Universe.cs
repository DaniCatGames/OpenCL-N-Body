namespace OpenCL_Barnes_Hut;

public enum UniverseSetups {
	EarthMoonSatellites,
	Infinity,
	LiLiaoUE2D
}

public abstract class Universe {
	private const double G = 6.674315e-11;

	public static (double[] positions, double[] velocities, double[] masses) GetUniverse(int numberOfBodies,
		UniverseSetups universeSetup, IntegrationMethods integrationMethod, double timeStep = 0) {

		return universeSetup switch {
			UniverseSetups.EarthMoonSatellites => EarthMoonSatellites(numberOfBodies, timeStep, integrationMethod),
			_ => Periodic(timeStep, integrationMethod, universeSetup),
		};
	}

	// Create initial data
	private static (double[] positions, double[] accelerations, double[] masses) EarthMoonSatellites(int numberOfBodies,
		double timeStep, IntegrationMethods integrationMethod) {
		var positions = new Double4[numberOfBodies];
		var velocities = new Double4[numberOfBodies];
		var masses = new double[numberOfBodies];

		masses[0] = 5.97219e24 * G; // Mass Earth
		masses[1] = 7.348e22 * G; // Mass Moon 
		positions[0] = new Double4(0.0); //Position Earth
		positions[1] =
			new Double4(-3.679525311419403E+08, 1.665317952836706E+08, 2.517731761270612E+07); //Position Moon
		velocities[0] = new Double4(0.0);
		velocities[1] = new Double4(-4.097678496220849E+02, -8.756270441426266E+02, -5.926517055394465E+01);

		for (var i = 2; i < numberOfBodies; i++) {
			masses[i] = 1200;
			positions[i] = new Double4(6571000 + i * 1000, 0.0, 0.0);
			velocities[i] = new Double4(0.0, 7800 + 3386 * (i / (float)numberOfBodies), 0.0);
		}

		if (timeStep == 0) return (FlattenDoubleArray(positions), FlattenDoubleArray(velocities), masses);
		
		return integrationMethod switch {
			IntegrationMethods.M52 => M52Init(numberOfBodies, timeStep, positions, velocities, masses),
			IntegrationMethods.M157 => M157Init(numberOfBodies, timeStep, positions, velocities, masses),
			_ => M52Init(numberOfBodies, timeStep, positions, velocities, masses)
		};
	}

	private static (double[] positions, double[] velocities, double[] masses) Periodic(double timeStep,
		IntegrationMethods integrationMethod, UniverseSetups universeSetup) {
		var numberOfBodies = 3;
		var positions = PeriodicOrbitData.PositionLookupTable()[universeSetup];
		var velocities = PeriodicOrbitData.VelocityLookupTable()[universeSetup];
		var masses = PeriodicOrbitData.MassLookupTable()[universeSetup];

		if (timeStep == 0) return (FlattenDoubleArray(positions), FlattenDoubleArray(velocities), masses);
		
		return integrationMethod switch {
			IntegrationMethods.M52 => M52Init(numberOfBodies, timeStep, positions, velocities, masses),
			IntegrationMethods.M157 => M157Init(numberOfBodies, timeStep, positions, velocities, masses),
			_ => M52Init(numberOfBodies, timeStep, positions, velocities, masses)
		};
	}
	
	
	// Multistep Initialization
	private static (double[] positions, double[] accelerations, double[] masses) M52Init(int numberOfBodies,
		double timeStep, Double4[] positions, Double4[] velocities, double[] masses) {
		//Calculate previous positions with Leapfrog using timesteps of dt/512
		var positions1 = CalculatePositionsAtTime(1, timeStep, positions, velocities, masses, numberOfBodies);
		var positions2 = CalculatePositionsAtTime(2, timeStep, positions, velocities, masses, numberOfBodies);
		var positions3 = CalculatePositionsAtTime(3, timeStep, positions, velocities, masses, numberOfBodies);

		var accelerations1 = new Double4[numberOfBodies];
		var accelerations2 = new Double4[numberOfBodies];

		for (var i = 0; i < numberOfBodies; i++)
			accelerations1[i] = CalculateTotalAcceleration(positions1, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations2[i] = CalculateTotalAcceleration(positions2, masses, numberOfBodies, i);

		var positionsArray = new Double4[numberOfBodies * 5];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i] = new Double4(0);
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 1] = positions[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 2] = positions1[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 3] = positions2[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 4] = positions3[i];

		var accelerationsArray = new Double4[numberOfBodies * 3];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i] = new Double4(0);
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 1] = accelerations1[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 2] = accelerations2[i];


		return (FlattenDoubleArray(positionsArray), FlattenDoubleArray(accelerationsArray), masses);
	}

	private static (double[] positions, double[] accelerations, double[] masses) M157Init(int numberOfBodies,
		double timeStep, Double4[] positions, Double4[] velocities, double[] masses) {
		//Calculate previous positions with Leapfrog using timesteps of dt/512
		var positions1 = CalculatePositionsAtTime(1, timeStep, positions, velocities, masses, numberOfBodies);
		var positions2 = CalculatePositionsAtTime(2, timeStep, positions, velocities, masses, numberOfBodies);
		var positions3 = CalculatePositionsAtTime(3, timeStep, positions, velocities, masses, numberOfBodies);
		var positions4 = CalculatePositionsAtTime(4, timeStep, positions, velocities, masses, numberOfBodies);
		var positions5 = CalculatePositionsAtTime(5, timeStep, positions, velocities, masses, numberOfBodies);
		var positions6 = CalculatePositionsAtTime(6, timeStep, positions, velocities, masses, numberOfBodies);
		var positions7 = CalculatePositionsAtTime(7, timeStep, positions, velocities, masses, numberOfBodies);
		var positions8 = CalculatePositionsAtTime(8, timeStep, positions, velocities, masses, numberOfBodies);
		var positions9 = CalculatePositionsAtTime(9, timeStep, positions, velocities, masses, numberOfBodies);
		var positions10 = CalculatePositionsAtTime(10, timeStep, positions, velocities, masses, numberOfBodies);
		var positions11 = CalculatePositionsAtTime(11, timeStep, positions, velocities, masses, numberOfBodies);
		var positions12 = CalculatePositionsAtTime(12, timeStep, positions, velocities, masses, numberOfBodies);
		var positions13 = CalculatePositionsAtTime(13, timeStep, positions, velocities, masses, numberOfBodies);

		var accelerations1 = new Double4[numberOfBodies];
		var accelerations2 = new Double4[numberOfBodies];
		var accelerations3 = new Double4[numberOfBodies];
		var accelerations4 = new Double4[numberOfBodies];
		var accelerations5 = new Double4[numberOfBodies];
		var accelerations6 = new Double4[numberOfBodies];
		var accelerations7 = new Double4[numberOfBodies];
		var accelerations8 = new Double4[numberOfBodies];
		var accelerations9 = new Double4[numberOfBodies];
		var accelerations10 = new Double4[numberOfBodies];
		var accelerations11 = new Double4[numberOfBodies];
		var accelerations12 = new Double4[numberOfBodies];

		for (var i = 0; i < numberOfBodies; i++)
			accelerations1[i] = CalculateTotalAcceleration(positions1, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations2[i] = CalculateTotalAcceleration(positions2, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations3[i] = CalculateTotalAcceleration(positions3, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations4[i] = CalculateTotalAcceleration(positions4, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations5[i] = CalculateTotalAcceleration(positions5, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations6[i] = CalculateTotalAcceleration(positions6, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations7[i] = CalculateTotalAcceleration(positions7, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations8[i] = CalculateTotalAcceleration(positions8, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations9[i] = CalculateTotalAcceleration(positions9, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations10[i] = CalculateTotalAcceleration(positions10, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations11[i] = CalculateTotalAcceleration(positions11, masses, numberOfBodies, i);
		for (var i = 0; i < numberOfBodies; i++)
			accelerations12[i] = CalculateTotalAcceleration(positions12, masses, numberOfBodies, i);

		var positionsArray = new Double4[numberOfBodies * 15];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i] = new Double4(0);
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 1] = positions[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 2] = positions1[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 3] = positions2[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 4] = positions3[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 5] = positions4[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 6] = positions5[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 7] = positions6[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 8] = positions7[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 9] = positions8[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 10] = positions9[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 11] = positions10[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 12] = positions11[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 13] = positions12[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 14] = positions13[i];

		var accelerationsArray = new Double4[numberOfBodies * 13];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i] = new Double4(0);
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 1] = accelerations1[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 2] = accelerations2[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 3] = accelerations3[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 4] = accelerations4[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 5] = accelerations5[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 6] = accelerations6[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 7] = accelerations7[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 8] = accelerations8[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 9] = accelerations9[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 10] = accelerations10[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 11] = accelerations11[i];
		for (var i = 0; i < numberOfBodies; i++) accelerationsArray[i + numberOfBodies * 12] = accelerations12[i];


		return (FlattenDoubleArray(positionsArray), FlattenDoubleArray(accelerationsArray), masses);
	}

	private static double[] FlattenDoubleArray(Double4[] array) {
		var flattened = new double[array.Length * 4];
		for (var i = 0; i < array.Length; i++) {
			flattened[i * 4] = array[i].X;
			flattened[i * 4 + 1] = array[i].Y;
			flattened[i * 4 + 2] = array[i].Z;
			flattened[i * 4 + 3] = array[i].W;
		}

		return flattened;
	}

	/// <summary>
	///     time 1 = p-1, time 2 = p-1, etc.
	/// </summary>
	private static Double4[] CalculatePositionsAtTime(double time, double timeStep, Double4[] positions,
		Double4[] velocities,
		double[] masses, int numberOfBodies) {
		var positionsAtTime = new Double4[numberOfBodies];
		var velocitiesAtTime = new Double4[numberOfBodies];

		for (var i = 0; i < numberOfBodies; i++) positionsAtTime[i] = positions[i];

		for (var i = 0; i < numberOfBodies; i++) velocitiesAtTime[i] = velocities[i];

		var calcTimeStep = timeStep / 512;
		var iterations = time * timeStep / calcTimeStep;

		//Kickoff 
		for (var i = 0; i < numberOfBodies; i++)
			velocitiesAtTime[i] +=
				-calcTimeStep * 0.5 * CalculateTotalAcceleration(positionsAtTime, masses, numberOfBodies, i);

		//Leapfrog
		for (var i = 0; i < iterations; i++) {
			for (var j = 0; j < numberOfBodies; j++)
				positionsAtTime[j] += -calcTimeStep * velocitiesAtTime[j];

			for (var j = 0; j < numberOfBodies; j++)
				velocitiesAtTime[j] +=
					-calcTimeStep * CalculateTotalAcceleration(positionsAtTime, masses, numberOfBodies, j);
		}

		return positionsAtTime;
	}

	private static Double4
		CalculateTotalAcceleration(Double4[] positions, double[] masses, int numberOfBodies, int id) {
		var acceleration = new Double4(0);

		for (var i = 0; i < numberOfBodies; i++)
			if (i != id)
				acceleration += CalculateAcceleration(positions[id], positions[i], masses[i]);

		return acceleration;
	}

	private static Double4 CalculateAcceleration(Double4 position1, Double4 position2, double mass2) {
		var difference = position2 - position1;
		var distanceSquared = difference.X * difference.X + difference.Y * difference.Y + difference.Z * difference.Z;
		return mass2 / (distanceSquared * double.Sqrt(distanceSquared)) * difference;
	}
}