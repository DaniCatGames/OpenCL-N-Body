namespace OpenCL_Barnes_Hut;

public struct Double4 {
	public readonly double X;
	public readonly double Y;
	public readonly double Z;
	public readonly double W;

	public Double4(double x, double y, double z) {
		X = x;
		Y = y;
		Z = z;
		W = 0.0;
	}

	public Double4(double x) {
		X = x;
		Y = x;
		Z = x;
		W = 0.0;
	}

	public Double4(double x, double y) {
		X = x;
		Y = y;
		Z = 0.0;
		W = 0.0;
	}

	private Double4(double x, double y, double z, double w) {
		X = x;
		Y = y;
		Z = z;
		W = w;
	}

	public static Double4 operator +(Double4 a) {
		return a;
	}

	public static Double4 operator -(Double4 a) {
		return new Double4(-a.X, -a.Y, -a.Z, -a.W);
	}

	public static Double4 operator +(Double4 a, Double4 b) {
		return new Double4(a.X + b.X, a.Y + b.Y, a.Z + b.Z, a.W + b.W);
	}

	public static Double4 operator -(Double4 a, Double4 b) {
		return new Double4(a.X - b.X, a.Y - b.Y, a.Z - b.Z, a.W - b.W);
	}

	public static Double4 operator *(Double4 a, Double4 b) {
		return new Double4(a.X * b.X, a.Y * b.Y, a.Z * b.Z, a.W * b.W);
	}

	public static Double4 operator /(Double4 a, Double4 b) {
		if (b.X == 0 || b.Y == 0 || b.Z == 0 || b.W == 0) throw new DivideByZeroException();
		return new Double4(a.X / b.X, a.Y / b.Y, a.Z / b.Z, a.W / b.W);
	}

	public static Double4 operator *(double a, Double4 b) {
		return new Double4(a * b.X, a * b.Y, a * b.Z, a * b.W);
	}
}