namespace MoMi;
class Blueprint
{
	private string methodName;
	private string className;
	private List<string> methodCalls;
	private List<string> variableNames;
	
	public Blueprint(string methodName, string className, List<string> methodCalls, List<string> variableNames)
	{
		this.methodName = methodName;
		this.className = className;
		this.methodCalls = methodCalls;
		this.variableNames = variableNames;
	}
	
	public void PrintBlueprint()
	{
		Console.WriteLine($"Method: {this.methodName}");
		Console.WriteLine($"Class: {this.className}");
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
	
	public void printMethodDependency()
	{
		Dictionary<string, List<string>> graph = new Dictionary<string, List<string>>();
		
		graph[this.methodName] = this.methodCalls;
		
		foreach (var pair in graph)
		{
			Console.WriteLine(pair.Key + ": " + String.Join(", ", pair.Value));
		}
	}
	
	public string getMethodName()
	{
		return this.methodName;
	}
	
	public List<string> getMethodCalls()
	{
		return this.methodCalls;
	}
}