namespace MoMi;
class Blueprint
{
	private string methodName;
	private string className;
	private string namespaceName;
	private List<string> methodCalls;
	private List<string> variableNames;
	
	public Blueprint(string methodName, string className, string namespaceName, List<string> methodCalls, List<string> variableNames)
	{
		this.methodName = methodName;
		this.className = className;
		this.namespaceName = namespaceName;
		this.methodCalls = methodCalls;
		this.variableNames = variableNames;
	}
	
	public void PrintBlueprint()
	{
		Console.WriteLine($"Method: {this.methodName}");
		Console.WriteLine($"Class: {this.className}");
		Console.WriteLine($"Namespace: {this.namespaceName}");
		Console.WriteLine($"Invocations:");
		foreach (string method in methodCalls)
		{
			Console.WriteLine($"-> {method}");
		}
		Console.WriteLine($"Variables:");
		foreach (string variable in variableNames)
		{
			Console.WriteLine($"-> {variable}");
		}
		Console.WriteLine("===========================================");
	}
}