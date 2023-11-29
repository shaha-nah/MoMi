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
import org.json.JSONObject;

import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.Problem;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.MethodCallExpr;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

public class Kowalski 
{
    public static void main( String[] args )
    {

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

        System.out.println( "Analysis Complete");
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
                String filePath = javaFile.getPath();
                String projectRootFolder = projectFolderPath.substring(projectFolderPath.lastIndexOf("/") + 1);
                String firstFolderAfterRoot = filePath.substring(filePath.indexOf(projectRootFolder) + projectRootFolder.length() + 1);
                String folder = firstFolderAfterRoot.substring(0, firstFolderAfterRoot.indexOf("/"));

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
                        methodData.put("Folder", folder);
                        methodData.put("ClassName", className);
                        methodData.put("MethodName", methodDeclaration.getNameAsString());

                        // Extract parameters
                        List<Parameter> parameters = methodDeclaration.getParameters();
                        List<String> parameterNames = new ArrayList<>();
                        for (Parameter parameter : parameters) {
                            parameterNames.add(parameter.getNameAsString());
                        }
                        methodData.put("Parameters", parameterNames);

                        // Extract comments
                        Optional<Comment> commentOptional = methodDeclaration.getComment();
                        if (commentOptional.isPresent()) {
                            Comment comment = commentOptional.get();
                            methodData.put("Comments", comment.getContent());
                        } else {
                            methodData.put("Comments", "");
                        }

                        // Extract AST features
                        List<String> astFeatures = extractASTFeatures(methodDeclaration);
                        methodData.put("ASTFeatures", astFeatures);

                        // Extract source code of the method
                        String methodSourceCode = methodDeclaration.toString();
                        methodData.put("MethodSourceCode", methodSourceCode);

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
                        // methodData.put("Folder", folderName);
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

    private static List<String> extractASTFeatures(MethodDeclaration methodDeclaration) {
        final List<String>[] astFeatures = new List[]{new ArrayList<>()};
    
        methodDeclaration.accept(new VoidVisitorAdapter<Void>() {
            @Override
            public void visit(MethodDeclaration node, Void arg) {
                // Add node information to AST features
                astFeatures[0].add(node.getClass().getSimpleName());
                super.visit(node, arg);
            }
        }, null);
    
        return astFeatures[0];
    }

}