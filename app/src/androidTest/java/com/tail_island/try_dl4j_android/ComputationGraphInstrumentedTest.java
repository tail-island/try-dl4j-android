package com.tail_island.try_dl4j_android;

import android.content.res.Resources;
import android.support.test.InstrumentationRegistry;
import android.support.test.runner.AndroidJUnit4;

import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.preprocessors.TensorFlowCnnToFeedForwardPreProcessor;
import org.deeplearning4j.util.ModelSerializer;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
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

import static org.junit.Assert.*;

@RunWith(AndroidJUnit4.class)
public class ComputationGraphInstrumentedTest {
    private INDArray input1;
    private INDArray input2;
    private INDArray input3;
    private INDArray input4;

    @Before
    public void loadInputs() throws IOException, ClassNotFoundException {
        Resources resources = InstrumentationRegistry.getTargetContext().getResources();

        try (InputStream stream = resources.openRawResource(R.raw.input1)) {
            input1 = (INDArray) new ObjectInputStream(stream).readObject();
        }

        try (InputStream stream = resources.openRawResource(R.raw.input2)) {
            input2 = (INDArray) new ObjectInputStream(stream).readObject();
        }

        try (InputStream stream = resources.openRawResource(R.raw.input3)) {
            input3 = (INDArray) new ObjectInputStream(stream).readObject();
        }

        try (InputStream stream = resources.openRawResource(R.raw.input4)) {
            input4 = (INDArray) new ObjectInputStream(stream).readObject();
        }
    }

    @Test
    public void testWithComputationGraph() throws Exception {
        Resources resources = InstrumentationRegistry.getTargetContext().getResources();

        // Adding reference. Because the model is imported from Keras and Reflections does not work on Android...
        NeuralNetConfiguration.reinitMapperWithSubtypes(Arrays.asList(new NamedType(TensorFlowCnnToFeedForwardPreProcessor.class)));

        ComputationGraph model;
        try (InputStream stream = resources.openRawResource(R.raw.model)) {
            model = ModelSerializer.restoreComputationGraph(stream);
        }

        INDArray output = model.outputSingle(input1, input2, input3, input4);

        assertEquals(0.02f, output.getFloat(0), 0.01f);
        assertEquals(0.34f, output.getFloat(1), 0.01f);
    }

    @Test
    public void testWithComputationGraphWrapper() throws Exception {
        Resources resources = InstrumentationRegistry.getTargetContext().getResources();

        // Adding reference. Because the model is imported from Keras and Reflections does not work on Android...
        NeuralNetConfiguration.reinitMapperWithSubtypes(Arrays.asList(new NamedType(TensorFlowCnnToFeedForwardPreProcessor.class)));

        ComputationGraph model;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(resources.openRawResource(R.raw.configuration)))) {
            StringBuilder sb = new StringBuilder();

            String line;
            while ((line = reader.readLine()) != null) {
                sb.append(line).append("\n");
            }

            model = new ComputationGraphWrapper(ComputationGraphConfiguration.fromJson(sb.toString()));
        }
        try (InputStream stream = resources.openRawResource(R.raw.coefficients)) {
            model.init(Nd4j.read(new DataInputStream(new BufferedInputStream(stream))), false);
        }

        INDArray output = model.outputSingle(input1, input2, input3, input4);

        assertEquals(0.02f, output.getFloat(0), 0.01f);
        assertEquals(0.34f, output.getFloat(1), 0.01f);
    }
}
