package ro.edu.aws.sgm.inference.pmml.randomforest.handler;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.xml.bind.JAXBException;
import javax.xml.transform.sax.SAXSource;


import org.dmg.pmml.FieldName;
import org.dmg.pmml.MiningModel;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.MiningModelEvaluator;
import org.jpmml.evaluator.ModelEvaluator;
import org.jpmml.evaluator.ProbabilityClassificationMap;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.springframework.stereotype.Service;
import org.xml.sax.InputSource;
import org.xml.sax.SAXException;

import ro.edu.aws.sgm.inference.pmml.randomforest.pojo.Features;

@Service("jpmmlInferenceImpl")
public class JPMMLInferenceHandlerImpl implements InferenceHandlerInf {

    public String predict(List <Features> data, Object model){

        File modelFile = (File)model;

        PMML pmmlFile = null;
        try {
          pmmlFile = createPMMLfromFile(modelFile);
        } catch (SAXException | IOException | JAXBException e) {
          e.printStackTrace();
        }
        List <Features> featuresList = data;

        StringBuilder sb = new StringBuilder();

        for( Features feature: featuresList){

            List <String> featureString = feature.getFeatures();
            String features = String.join(",", featureString).concat("\n");
            sb.append(features);
         }

        ModelEvaluator<MiningModel> modelEvaluator = new MiningModelEvaluator(pmmlFile);

        return predict(sb.toString().lines(), modelEvaluator);
    
    }
    

private static String predict(Stream<String> inputData,
      ModelEvaluator<MiningModel> modelEvaluator) {


    String returns = inputData.map(dataLine -> {

      System.out.println("Predicting for input data: " + dataLine);
      Map<FieldName, FieldValue> arguments = readArgumentsFromLine(dataLine, modelEvaluator);
      modelEvaluator.verify();
      Map<FieldName, ?> results = modelEvaluator.evaluate(arguments);
      FieldName targetName = modelEvaluator.getTargetField();
      Object targetValue = results.get(targetName);
      ProbabilityClassificationMap nodeMap = (ProbabilityClassificationMap) targetValue;

      return  ( nodeMap != null && nodeMap.getResult() !=  null) ? nodeMap.getResult().toString() : "NA for input->"+dataLine;
    }).collect(Collectors.joining(System.lineSeparator()));

    System.out.println("Prediction results: " + returns);
    return returns;

  }

  private static Map<FieldName, FieldValue> readArgumentsFromLine(String line,
      ModelEvaluator<MiningModel> modelEvaluator) {
    Map<FieldName, FieldValue> arguments = new LinkedHashMap<FieldName, FieldValue>();
    String[] lineArgs = line.split(",");

    if (lineArgs.length != 5)
      return arguments;

    FieldValue sepalLength = modelEvaluator.prepare(new FieldName("Sepal.Length"),
        lineArgs[0].isEmpty() ? 0 : lineArgs[0]);
    FieldValue sepalWidth = modelEvaluator.prepare(new FieldName("Sepal.Width"),
        lineArgs[1].isEmpty() ? 0 : lineArgs[1]);
    FieldValue petalLength = modelEvaluator.prepare(new FieldName("Petal.Length"),
        lineArgs[2].isEmpty() ? 0 : lineArgs[2]);
    FieldValue petalWidth = modelEvaluator.prepare(new FieldName("Petal.Width"),
        lineArgs[3].isEmpty() ? 0 : lineArgs[3]);

    arguments.put(new FieldName("Sepal.Length"), sepalLength);
    arguments.put(new FieldName("Sepal.Width"), sepalWidth);
    arguments.put(new FieldName("Petal.Length"), petalLength);
    arguments.put(new FieldName("Petal.Width"), petalWidth);

    return arguments;
  }

 

private static PMML createPMMLfromFile(File pmmlFile)
  throws SAXException, IOException, JAXBException{


 // File pmmlFile = new File(SGMController.class.getResource(fileName).getPath());  
  String pmmlString = new Scanner(pmmlFile).useDelimiter("\\Z").next();

  InputStream is = new ByteArrayInputStream(pmmlString.getBytes());

  InputSource source = new InputSource(is);
  SAXSource transformedSource = ImportFilter.apply(source);

  return JAXBUtil.unmarshalPMML(transformedSource);

}
}
