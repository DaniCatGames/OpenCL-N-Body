﻿namespace OpenCL_Barnes_Hut;

public enum UniverseSetup {
	EarthMoonSatellites,
	SunSystem
}

public class Universe {
	private const double G = 6.674315e-11;

	public static (double[] positions, double[] velocities, double[] masses) GetUniverse(int numberOfBodies,
		UniverseSetup setup) {
		return setup switch {
			UniverseSetup.EarthMoonSatellites => EarthMoonSatellites(numberOfBodies),
			UniverseSetup.SunSystem => SunSystem(),
			_ => EarthMoonSatellites(numberOfBodies)
		};
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
}