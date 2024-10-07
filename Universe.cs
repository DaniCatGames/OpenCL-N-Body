namespace OpenCL_Barnes_Hut;

public enum UniverseSetups {
	EarthMoonSatellites,
	SunSystem
}

public class Universe {
	private const double G = 6.674315e-11;

	public static (double[] positions, double[] velocities, double[] masses) GetUniverse(int numberOfBodies,
		UniverseSetups setups, double timeStep = 0) {
		if (timeStep == 0)
			return setups switch {
				UniverseSetups.EarthMoonSatellites => EarthMoonSatellites(numberOfBodies),
				UniverseSetups.SunSystem => SunSystem(),
				_ => EarthMoonSatellites(numberOfBodies)
			};

		return setups switch {
			UniverseSetups.EarthMoonSatellites => EarthMoonSatellites(numberOfBodies, timeStep),
			UniverseSetups.SunSystem => SunSystem(),
			_ => EarthMoonSatellites(numberOfBodies, timeStep)
		};
	}

	private static (double[] positions, double[] velocities, double[] masses) EarthMoonSatellites(int numberOfBodies,
		double timeStep) {
		var positions = new Double4[numberOfBodies];
		var velocities = new Double4[numberOfBodies];
		var masses = new double[numberOfBodies];

		masses[0] = 5.972e24 * G; // Mass Earth
		masses[1] = 7.348e22 * G; // Mass Moon 
		positions[0] = new Double4(0.0); //Position Earth
		positions[1] = new Double4(3.84e8, 0.0, 0.0); //Position Moon
		velocities[0] = new Double4(0.0);
		velocities[1] = new Double4(0.0, 1022.0, 0.0);

		for (var i = 2; i < numberOfBodies; i++) {
			masses[i] = 1200;
			positions[i] = new Double4(6571000 + i * 1000, 0.0, 0.0);
			velocities[i] = new Double4(0.0, 7800 + 3386 * (i / (float)numberOfBodies), 0.0);
		}

		//Calculate previous positions with Leapfrog using timesteps of dt/512
		var positions1 = CalculatePositionsAtTime(1, timeStep, positions, velocities, masses, numberOfBodies);
		var positions2 = CalculatePositionsAtTime(2, timeStep, positions, velocities, masses, numberOfBodies);
		var positions3 = CalculatePositionsAtTime(3, timeStep, positions, velocities, masses, numberOfBodies);

		var positionsArray = new Double4[numberOfBodies * 5];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i] = new Double4(0);
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies] = positions[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 2] = positions1[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 3] = positions2[i];
		for (var i = 0; i < numberOfBodies; i++) positionsArray[i + numberOfBodies * 4] = positions3[i];

		return (FlattenDoubleArray(positionsArray), FlattenDoubleArray(velocities), masses);
	}

	private static (double[] positions, double[] velocities, double[] masses) EarthMoonSatellites(int numberOfBodies) {
		var positions = new Double4[numberOfBodies];
		var velocities = new Double4[numberOfBodies];
		var masses = new double[numberOfBodies];

		masses[0] = 5.972e24 * G; // Mass Earth
		masses[1] = 7.348e22 * G; // Mass Moon 
		positions[0] = new Double4(0.0); //Position Earth
		positions[1] = new Double4(3.84e8, 0.0, 0.0); //Position Moon
		velocities[0] = new Double4(0.0);
		velocities[1] = new Double4(0.0, 1022.0, 0.0);

		for (var i = 2; i < numberOfBodies; i++) {
			masses[i] = 1200;
			positions[i] = new Double4(6571000 + i * 1000, 0.0, 0.0);
			velocities[i] = new Double4(0.0, 7800 + 3386 * (i / (float)numberOfBodies), 0.0);
		}

		var flattenedPositions = FlattenDoubleArray(positions);
		var flattenedVelocities = FlattenDoubleArray(velocities);

		return (flattenedPositions, flattenedVelocities, masses);
	}

	private static (double[] positions, double[] velocities, double[] masses) SunSystem() {
		var positions = new Double4[9];
		var velocities = new Double4[9];
		var masses = new double[9];


		return (FlattenDoubleArray(positions), FlattenDoubleArray(velocities), masses);
	}

	private static double[] FlattenDoubleArray(Double4[] array) {
		var flattened = new double[array.Length * 4];
		for (var i = 0; i < array.Length; i++) {
			flattened[i * 4] = array[i].X;
			flattened[i * 4 + 1] = array[i].Y;
			flattened[i * 4 + 2] = array[i].Z;
			flattened[i * 4 + 2] = array[i].W;
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
		var iterations = time / calcTimeStep;

		//Kickoff 
		for (var i = 0; i < numberOfBodies; i++)
			velocitiesAtTime[i] +=
				-calcTimeStep * 0.5 * CalculateAcceleration(positionsAtTime, masses, numberOfBodies, i);

		//Leapfrog
		for (var i = 0; i < iterations; i++) {
			for (var j = 0; j < numberOfBodies; j++)
				positionsAtTime[j] += -calcTimeStep * velocitiesAtTime[j];

			for (var j = 0; j < numberOfBodies; j++)
				velocitiesAtTime[j] +=
					-calcTimeStep * CalculateAcceleration(positionsAtTime, masses, numberOfBodies, j);
		}

		return positionsAtTime;
	}

	private static Double4 CalculateAcceleration(Double4[] positions, double[] masses, int numberOfBodies, int id) {
		var acceleration = new Double4(0);

		for (var i = 0; i < numberOfBodies; i++)
			if (i != id)
				acceleration += ComputeAcceleration(positions[id], positions[i], masses[i]);

		return acceleration;
	}

	private static Double4 ComputeAcceleration(Double4 position1, Double4 position2, double mass2) {
		var difference = position2 - position1;
		var distanceSquared = difference.X * difference.X + difference.Y * difference.Y + difference.Z * difference.Z;
		return mass2 / (distanceSquared * double.Sqrt(distanceSquared)) * difference;
	}
}