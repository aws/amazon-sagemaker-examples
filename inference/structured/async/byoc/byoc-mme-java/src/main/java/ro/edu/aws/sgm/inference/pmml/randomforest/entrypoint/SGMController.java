package ro.edu.aws.sgm.inference.pmml.randomforest.entrypoint;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestMethod;
import org.springframework.web.bind.annotation.RestController;

import jakarta.annotation.PostConstruct;
import jakarta.servlet.http.HttpServletRequest;
import ro.edu.aws.sgm.inference.pmml.randomforest.exception.InsufficientMemoryException;
import ro.edu.aws.sgm.inference.pmml.randomforest.exception.ModelAlreadyPresentException;
import ro.edu.aws.sgm.inference.pmml.randomforest.exception.ModelNotFoundException;
import ro.edu.aws.sgm.inference.pmml.randomforest.handler.InferenceHandlerInf;
import ro.edu.aws.sgm.inference.pmml.randomforest.pojo.Features;
import ro.edu.aws.sgm.inference.pmml.randomforest.pojo.InputData;
import ro.edu.aws.sgm.inference.pmml.randomforest.pojo.MemoryStats;
import ro.edu.aws.sgm.inference.pmml.randomforest.pojo.Model;

@RestController
public class SGMController {

  private ConcurrentHashMap<String, File> concurrentHashMap;

  @Autowired
  @Qualifier("jpmmlInferenceImpl")
  private InferenceHandlerInf jpmmlInferenceHandlerImpl;

  @Value("${java.memory.threshold.percentage}")
  private String memoryThresholdValue;

  @PostConstruct
  public void init() {
   
      concurrentHashMap = new ConcurrentHashMap<String, File>();
   
  }

  @RequestMapping(value = "/ping", method = RequestMethod.GET)
  public String ping() {

    System.out.println("[Ping] Pinging the container.");
    return "";
  }


  @PostMapping(value = "/models/{model_name}/invoke")
  public ResponseEntity<String> invoke(HttpServletRequest request, 
    @RequestHeader(value = "X-Amzn-SageMaker-Target-Model") String targetModel,
    @PathVariable String model_name, @RequestBody InputData inputData) throws IOException {
    
    System.out.println("[Invoke Model] Target model header value: "+ targetModel);  
    String predictions = null;

    if(isModelLoaded(model_name)){
      
      System.out.println("[Invoke Model] Found model " +model_name+" in memory.");
      List <Features> data = inputData.getFeatureList();
      predictions = jpmmlInferenceHandlerImpl.predict(data, concurrentHashMap.get(model_name));
      
    }
 
    return new ResponseEntity <String>(predictions, HttpStatus.OK);
    
  }


  @PostMapping(value = "/models")
  public ResponseEntity <?>loadModel(@RequestBody Model model) throws Exception{

    String model_name = model.getModel_name();
    String url = model.getUrl();

    System.out.println("[Load Model] model_name: "+ model_name);
    System.out.println("[Load Model] url: "+ url);

    // Throw exception when model is already present in the Map
    if(concurrentHashMap.containsKey(model_name)){
      throw new ModelAlreadyPresentException("Model Name: " + model_name + "already loaded in memory.");
    }

    // Throw exception when enough memory is not available to load the mmodel
    if(!isMemoryAvailable()){
      throw new InsufficientMemoryException("Insufficient memory. Cannot load model: " + model_name);
    }

    File modelFile = Paths.get(url).toFile();

    concurrentHashMap.put(model_name, modelFile);
    System.out.println("[Load Model] model_name: "+ model_name + " loaded in memory.");

  
    return  new ResponseEntity<>("Model: "+ model_name + " loaded in memory", HttpStatus.OK);
  }


  @GetMapping("/models")
  public ResponseEntity<List<Model>> listModels(HttpServletRequest request) throws IOException{

    List <Model> modelList = new ArrayList<Model>();

    for (String key: concurrentHashMap.keySet()){

      Model model = new Model(key, "opt/ml/models/" +key+"/model");
      modelList.add(model);
    }

    System.out.println("[List Models] Returning a list of "+modelList.size()+" models.");
    return ResponseEntity.ok(modelList);
  }


  @GetMapping("/models/{model_name}")
  public ResponseEntity<Model> getModel(@PathVariable String model_name){

    Model model = null;
    if(isModelLoaded(model_name)){

      model = new Model(model_name, "opt/ml/models/" +model_name+"/model");
      System.out.println("[Get Model] Returning model name" +model.getModel_name()+" and Model URL " +model.getUrl());

    }
    return ResponseEntity.ok(model);
  }


  @DeleteMapping("/models/{model_name}")
  public ResponseEntity <?> deleteModel(@PathVariable String model_name){

    if(isModelLoaded(model_name)){
     concurrentHashMap.remove(model_name);
     System.out.println("[Delete Model] Deleting model: "+model_name+" from the memory");
    }

    return new ResponseEntity<String>("Model: " + model_name + " removed from the memory.", HttpStatus.OK);
  }

  @GetMapping("memory-status")
  public MemoryStats getMemoryStatistics() {
      MemoryStats stats = new MemoryStats();
      stats.setHeapSize(Runtime.getRuntime().totalMemory());
      stats.setHeapMaxSize(Runtime.getRuntime().maxMemory());
      stats.setHeapFreeSize(Runtime.getRuntime().freeMemory());
      return stats;
  }

  private boolean isModelLoaded(String model_name){

    if(!concurrentHashMap.containsKey(model_name)){
      throw new ModelNotFoundException("Model name: "+ model_name + "not loaded in memory");
    }
    return true;
  }

  private boolean isMemoryAvailable(){

    long freeMemory = Runtime.getRuntime().freeMemory();
    long maxMemory =  Runtime.getRuntime().maxMemory();

    System.out.println("Current runtime free memory: "+ freeMemory);
    System.out.println("Max memory available: "+maxMemory);

    double availableMemory = ((double)freeMemory/maxMemory) * 100;

    System.out.println("Current heap memory available: "+ availableMemory+"%");
    System.out.println("Memory threshold value: "+memoryThresholdValue+"%");

      if( availableMemory >= Integer.parseInt(memoryThresholdValue))
        return false;

    return true;

  }

 
}


