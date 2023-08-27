using System;

class Functions
{
	public void PrintMessage()
	{
		Console.WriteLine("Hello, world!");
	}
	public int AddNumbers(int a, int b)
	{
		return a + b;
	}

	public void runProgram()
	{
		PrintMessage();
		Console.WriteLine(AddNumbers(5, 6));
	}
}