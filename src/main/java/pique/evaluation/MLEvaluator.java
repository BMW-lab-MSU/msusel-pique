package pique.evaluation;

import java.math.BigDecimal;

import pique.evaluation.Evaluator;
import pique.model.ModelNode;
import pique.utility.BigDecimalWithContext;
import weka.classifiers.Classifier;

//TODO (1.0): Documentation
public class MLEvaluator extends Evaluator {

    @Override
    public BigDecimal evaluate(ModelNode inNode) {

        // TODO (1.0): Some redesign needed to better handle quality model description where there are not yet weights,
        //  values, etc...

        String nodeName = inNode.getName();
        BigDecimal outValue = new BigDecimalWithContext("0.0");


        if (inNode.getIndirectChildren() != null) {
            for (ModelNode child : inNode.getIndirectChildren().values()) {
                child.getValue();
            }
        } else {

        }

        // loading the model
        System.out.println("----------loading model---------------");
        try {
            Classifier loaded_model = (Classifier) weka.core.SerializationHelper.read("./lin.model");
        }
        catch (Exception e) {
            e.printStackTrace();
        }

//        // Apply weighted sums
//        for (ModelNode child : inNode.getChildren().values()) {
//            outValue = outValue.add(child.getValue().multiply(inNode.getWeight(child.getName()),BigDecimalWithContext.getMC()));
//        }

        return outValue;
    }
}
