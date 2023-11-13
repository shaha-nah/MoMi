package com.example.kowalski;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.filefilter.DirectoryFileFilter;
import org.apache.commons.io.filefilter.SuffixFileFilter;
import org.json.JSONArray;
import org.json.JSONObject;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.Problem;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.MethodCallExpr; 

public class Kowalski 
{
    public static void main( String[] args )
    {
        System.out.println( "Analysis");

        String[] projectFolderPaths = {
            "/media/shahanah/4311F5B9262F01CF1/MSc Software Project Management/Dissertation/code/MoMi/Population/spring-petclinic-microservices",
            "/media/shahanah/4311F5B9262F01CF1/MSc Software Project Management/Dissertation/code/MoMi/Population/microservices-event-sourcing",
            "/media/shahanah/4311F5B9262F01CF1/MSc Software Project Management/Dissertation/code/MoMi/Population/es-kanban-board/java-server"
        };
    
        String[] jsonFilePaths = {
            "/media/shahanah/4311F5B9262F01CF1/MSc Software Project Management/Dissertation/code/MoMi/Decomposition/inputs/petclinic.json",
            "/media/shahanah/4311F5B9262F01CF1/MSc Software Project Management/Dissertation/code/MoMi/Decomposition/inputs/event.json",
            "/media/shahanah/4311F5B9262F01CF1/MSc Software Project Management/Dissertation/code/MoMi/Decomposition/inputs/kanban.json"
        };
    
        for (int i = 0; i < projectFolderPaths.length; i++) {
            analysis(projectFolderPaths[i], jsonFilePaths[i]);
        }
        
    }

    private static void analysis(String projectFolderPath, String outputFileName){
        List<File> javaFiles = (List<File>) FileUtils.listFiles(
                new File(projectFolderPath),
                new SuffixFileFilter(".java"),
                DirectoryFileFilter.DIRECTORY
        );
        List outputData = new ArrayList<>();

        for (File javaFile : javaFiles) {
            ParseResult<CompilationUnit> result;
            try {
                result = new JavaParser().parse(javaFile);
                if (result.isSuccessful()) {
                    CompilationUnit compilationUnit = result.getResult().get();

                    List<JSONObject> methodDataList = new ArrayList<>();

                    List<MethodDeclaration> methodDeclarations = compilationUnit.findAll(MethodDeclaration.class);
                    String className;
                    Optional<ClassOrInterfaceDeclaration> classDeclarationOptional = compilationUnit.findFirst(ClassOrInterfaceDeclaration.class);
                    if (classDeclarationOptional.isPresent()) {
                        ClassOrInterfaceDeclaration classDeclaration = classDeclarationOptional.get();
                        className = classDeclaration.getNameAsString();
                    } else {
                        className = "";
                    }
                    for (MethodDeclaration methodDeclaration : methodDeclarations) {
                        JSONObject methodData = new JSONObject();
                        methodData.put("ClassName", className);
                        methodData.put("MethodName", methodDeclaration.getNameAsString());

                        List<JSONObject> variableDataList = new ArrayList<>();
                        List<VariableDeclarator> variableDeclarators = methodDeclaration.findAll(VariableDeclarator.class);
                        for (VariableDeclarator variableDeclarator : variableDeclarators) {
                            JSONObject variableData = new JSONObject();
                            variableData.put("VariableName", variableDeclarator.getNameAsString());
                            variableDataList.add(variableData);
                        }
                        methodData.put("Variables", variableDataList);

                        List<JSONObject> methodCallList = new ArrayList<>();
                        List<MethodCallExpr> methodCalls = methodDeclaration.findAll(MethodCallExpr.class);
                        for (MethodCallExpr methodCall : methodCalls) {
                            JSONObject methodCallData = new JSONObject();
                            if (isMethodDeclaredInProject(methodCall, methodDeclarations)) {
                                methodCallData.put("MethodCalled", methodCall.getNameAsString());
                                methodCallList.add(methodCallData);
                            }
                        }
                        methodData.put("Methodscalled", methodCallList);
                        if (methodData.length() != 0) {
                            methodDataList.add(methodData);
                        }
                    }

                    outputData.addAll(methodDataList);


                } else {
                    System.out.println("Parse errors occurred:");
                    List<Problem> problems = result.getProblems();
                    for (Problem problem : problems){
                        System.out.println(problem);
                    }
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        JSONObject jsonData = new JSONObject();
        jsonData.put("Data", outputData);
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(outputFileName))) {
            writer.write(jsonData.toString());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static boolean isMethodDeclaredInProject(MethodCallExpr methodCall, List<MethodDeclaration> methodDeclarations) {
        for (MethodDeclaration declaration : methodDeclarations) {
            if (declaration.getNameAsString().equals(methodCall.getNameAsString())) {
                return true;
            }
        }
        return false;
    }
}