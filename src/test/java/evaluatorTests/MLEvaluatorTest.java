package evaluatorTests;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.Test;
import pique.evaluation.MLEvaluator;
import pique.model.*;
import pique.runnable.ASingleProjectEvaluator;
import pique.utility.BigDecimalWithContext;
import utilities.PiqueTestProperties_afterBenchmarker;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Properties;

import java.math.BigDecimal;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

import org.junit.Test;

import pique.calibration.CSVWeighter;
import pique.calibration.MLWeighter;
import pique.calibration.IWeighter;
import pique.calibration.WeightResult;
import pique.model.QualityModel;
import pique.model.QualityModelImport;
import utilities.PiqueTestProperties;
import utilities.PiqueTestProperties_afterBenchmarker;


public class MLEvaluatorTest {



    @BeforeClass
    public static void setup(){

    }

//    @Test
//    public void testEvaluate(){
//        new ASingleProjectEvaluator("input/docker-image-target.json");
//    }

    @Test
    public void MLEvaluatorTest(){
        Properties prop = PiqueTestProperties_afterBenchmarker.getProperties();

        Path blankqmFilePath = Paths.get(prop.getProperty("blankqm.filepath"));
        String pathToCsv = prop.getProperty("benchmark.pathToCSV");

        QualityModelImport qmImport = new QualityModelImport(blankqmFilePath);
        QualityModel qmDescription = qmImport.importQualityModel();
        IWeighter weighter = new MLWeighter();
//        Set<WeightResult> results = weighter.elicitateWeights(qmDescription, Paths.get(pathToCsv));


        Map<String, ModelNode> qaNodes= qmDescription.getQualityAspects();

        Map<String, ModelNode> msNodes= qmDescription.getMeasures();
        msNodes.forEach((msNodeName,msNode)->{
            msNode.setValue(new BigDecimalWithContext(0.0));
        });


        qaNodes.forEach((qaNodeName,qaNode)->{
            System.out.println(qaNodeName+"\t"+qaNode);

            MLEvaluator qaEvaluator = new MLEvaluator();

            BigDecimal outValue = qaEvaluator.evaluate(qaNode);

        });


    }

    @AfterClass
    public static void cleanup(){

    }
}
