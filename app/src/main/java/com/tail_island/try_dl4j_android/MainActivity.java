package com.tail_island.try_dl4j_android;

import android.os.AsyncTask;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.TensorFlowCnnToFeedForwardPreProcessor;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.jackson.databind.jsontype.NamedType;

import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class MainActivity extends AppCompatActivity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        AsyncTask.execute(new Runnable() {
            @Override
            public void run() {
                try {
                    test1();
                    test3();

                } catch (Exception ex) {
                    Log.e("try-dl4j-android", ex.toString());
                }
            }
        });
    }

    private void test1() throws IOException, ClassNotFoundException {
        // The model is imported from Keras. So, it uses some extra classes. But Reflections library does not work on Android OS. I add extra classes by hand...
        NeuralNetConfiguration.reinitMapperWithSubtypes(Arrays.asList(new NamedType(TensorFlowCnnToFeedForwardPreProcessor.class)));

        ComputationGraph model = getModel(R.raw.model);

        INDArray input1 = getInput(R.raw.input1);
        INDArray input2 = getInput(R.raw.input2);
        INDArray input3 = getInput(R.raw.input3);
        INDArray input4 = getInput(R.raw.input4);

        INDArray output = model.outputSingle(input1, input2, input3, input4);

        System.out.println(output);
    }

    private void test2() throws IOException {
        NeuralNetConfiguration.reinitMapperWithSubtypes(Arrays.asList(new NamedType(TensorFlowCnnToFeedForwardPreProcessor.class)));
        ComputationGraphConfiguration configuration = ComputationGraphConfiguration.fromJson(getContent(R.raw.configuration));

        // I copied below code from org.deeplearning4j.nn.graph.ComputationGraph#topologicalSortOrder.

        //https://en.wikipedia.org/wiki/Topological_sorting#Kahn.27s_algorithm
        Map<String, org.deeplearning4j.nn.conf.graph.GraphVertex> nodeMap = configuration.getVertices();
        List<String> networkInputNames = configuration.getNetworkInputs();
        int numVertices = networkInputNames.size() + configuration.getVertices().size();
        int[] out = new int[numVertices];
        int outCounter = 0;

        //First: represent the graph more usefully as a Map<Integer,Set<Integer>>, where map represents edges i -> j
        // key represents j, set is set of i (inputs) for vertices j
        Map<Integer, String> vertexNamesMap = new HashMap<>();
        Map<String, Integer> vertexNamesMap2 = new HashMap<>();
        int i = 0;
        for (String inputName : configuration.getNetworkInputs()) {
            vertexNamesMap.put(i, inputName);
            vertexNamesMap2.put(inputName, i);
            i++;
        }
        for (Map.Entry<String, org.deeplearning4j.nn.conf.graph.GraphVertex> entry : nodeMap.entrySet()) {
            String name = entry.getKey();
            vertexNamesMap.put(i, name);
            vertexNamesMap2.put(name, i);
            i++;
        }

        Map<Integer, Set<Integer>> inputEdges = new HashMap<>(); //key: vertex. Values: vertices that the key vertex receives input from
        Map<Integer, Set<Integer>> outputEdges = new HashMap<>(); //key: vertex. Values: vertices that the key vertex outputs to

        for (String s : configuration.getNetworkInputs()) {
            int idx = vertexNamesMap2.get(s);
            inputEdges.put(idx, null);
        }

        for (Map.Entry<String, org.deeplearning4j.nn.conf.graph.GraphVertex> entry : nodeMap.entrySet()) {
            String thisVertexName = entry.getKey();
            int idx = vertexNamesMap2.get(thisVertexName);
            List<String> inputsToThisVertex = configuration.getVertexInputs().get(thisVertexName);

            if (inputsToThisVertex == null || inputsToThisVertex.isEmpty()) {
                inputEdges.put(idx, null);
                continue;
            }

            Set<Integer> inputSet = new HashSet<>();
            for (String s : inputsToThisVertex) {
                Integer inputIdx = vertexNamesMap2.get(s);
                if (inputIdx == null) {
                    System.out.println();
                }
                inputSet.add(inputIdx);
                Set<Integer> outputSetForInputIdx = outputEdges.get(inputIdx);
                if (outputSetForInputIdx == null) {
                    outputSetForInputIdx = new HashSet<>();
                    outputEdges.put(inputIdx, outputSetForInputIdx);
                }
                outputSetForInputIdx.add(idx); //input vertex outputs to the current vertex
            }

            // *** Printing putting order
            System.out.println(idx);

            inputEdges.put(idx, inputSet);
        }

        // *** Printing iterating order
        // *** On x86 putting order and iterating order is same. But on ARM, they are different...
        for (Map.Entry<Integer, Set<Integer>> entry : inputEdges.entrySet()) {
            System.out.println(entry.getKey());
        }

        //Now: do topological sort
        //Set of all nodes with no incoming edges: (this would be: input vertices)
        LinkedList<Integer> noIncomingEdges = new LinkedList<>();
        for (Map.Entry<Integer, Set<Integer>> entry : inputEdges.entrySet()) {
            Set<Integer> inputsFrom = entry.getValue();
            if (inputsFrom == null || inputsFrom.isEmpty()) {
                noIncomingEdges.add(entry.getKey());
            }
        }

        while (!noIncomingEdges.isEmpty()) {
            int next = noIncomingEdges.removeFirst();
            out[outCounter++] = next; //Add to sorted list

            Set<Integer> vertexOutputsTo = outputEdges.get(next);

            //Remove edges next -> vertexOuputsTo[...] from graph;
            if (vertexOutputsTo != null) {
                for (Integer v : vertexOutputsTo) {
                    Set<Integer> set = inputEdges.get(v);
                    set.remove(next);
                    if (set.isEmpty()) {
                        noIncomingEdges.add(v); //No remaining edges for vertex i -> add to list for processing
                    }
                }
            }
        }

        //If any edges remain in the graph: graph has cycles:
        for (Map.Entry<Integer, Set<Integer>> entry : inputEdges.entrySet()) {
            Set<Integer> set = entry.getValue();
            if (set == null)
                continue;
            if (!set.isEmpty())
                throw new IllegalStateException(
                        "Invalid configuration: cycle detected in graph. Cannot calculate topological ordering with graph cycle ("
                                + "cycle includes vertex \"" + vertexNamesMap.get(entry.getKey())
                                + "\")");
        }

        System.out.println(Arrays.toString(out));
    }

    private void test3() throws IOException, ClassNotFoundException {
        NeuralNetConfiguration.reinitMapperWithSubtypes(Arrays.asList(new NamedType(TensorFlowCnnToFeedForwardPreProcessor.class)));

        // *** Using ComputationGraphWrapper which overrides the topologicalSortOrder method.
        ComputationGraph model = new ComputationGraphWrapper(ComputationGraphConfiguration.fromJson(getContent(R.raw.configuration)));
        model.init(getCoefficients(R.raw.coefficients), false);

        INDArray input1 = getInput(R.raw.input1);
        INDArray input2 = getInput(R.raw.input2);
        INDArray input3 = getInput(R.raw.input3);
        INDArray input4 = getInput(R.raw.input4);

        INDArray output = model.outputSingle(input1, input2, input3, input4);

        System.out.println(output);
    }

    private INDArray getCoefficients(int resourceId) throws IOException, ClassNotFoundException {
        try (InputStream stream = getResources().openRawResource(resourceId)) {
            return Nd4j.read(new DataInputStream(stream));
        }
    }

    private ComputationGraph getModel(int resourceId) throws IOException {
        try (InputStream stream = getResources().openRawResource(resourceId)) {
            return ModelSerializer.restoreComputationGraph(stream);
        }
    }

    private INDArray getInput(int resourceId) throws IOException, ClassNotFoundException {
        try (InputStream stream = getResources().openRawResource(resourceId)) {
            return (INDArray)new ObjectInputStream(stream).readObject();
        }
    }

    private String getContent(int resourceId) throws IOException {
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(getResources().openRawResource(resourceId)))) {
            StringBuilder sb = new StringBuilder();

            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }

            return sb.toString();
        }
    }
}
