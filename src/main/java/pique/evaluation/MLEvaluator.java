package pique.evaluation;

import java.math.BigDecimal;

import pique.evaluation.Evaluator;
import pique.model.ModelNode;
import pique.utility.BigDecimalWithContext;

//TODO (1.0): Documentation
public class MLEvaluator extends Evaluator {

    @Override
    public BigDecimal evaluate(ModelNode inNode) {

        // TODO (1.0): Some redesign needed to better handle quality model description where there are not yet weights,
        //  values, etc...

        BigDecimal outValue = new BigDecimalWithContext("0.0");



        // Apply weighted sums
        for (ModelNode child : inNode.getChildren().values()) {
            outValue = outValue.add(child.getValue().multiply(inNode.getWeight(child.getName()),BigDecimalWithContext.getMC()));
        }

        return outValue;
    }
}
