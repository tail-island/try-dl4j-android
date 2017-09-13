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

import java.io.BufferedInputStream;
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
                    test2();

                } catch (Exception ex) {
                    Log.e("try-dl4j-android", ex.toString());
                }
            }
        });
    }

    private void test1() throws IOException, ClassNotFoundException {
        // Adding reference. Because the model is imported from Keras and Reflections does not work on Android...
        NeuralNetConfiguration.reinitMapperWithSubtypes(Arrays.asList(new NamedType(TensorFlowCnnToFeedForwardPreProcessor.class)));

        ComputationGraph model = getModel(R.raw.model);

        System.out.println(model.outputSingle(getInput(R.raw.input1), getInput(R.raw.input2), getInput(R.raw.input3), getInput(R.raw.input4)));
    }

    private void test2() throws IOException, ClassNotFoundException {
        NeuralNetConfiguration.reinitMapperWithSubtypes(Arrays.asList(new NamedType(TensorFlowCnnToFeedForwardPreProcessor.class)));

        // *** Using ComputationGraphWrapper that overrides the topologicalSortOrder method.
        ComputationGraph model = new ComputationGraphWrapper(ComputationGraphConfiguration.fromJson(getContent(R.raw.configuration)));
        model.init(getCoefficients(R.raw.coefficients), false);

        System.out.println(model.outputSingle(getInput(R.raw.input1), getInput(R.raw.input2), getInput(R.raw.input3), getInput(R.raw.input4)));
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

    private INDArray getCoefficients(int resourceId) throws IOException, ClassNotFoundException {
        try (InputStream stream = getResources().openRawResource(resourceId)) {
            return Nd4j.read(new DataInputStream(new BufferedInputStream(stream)));
        }
    }
}
