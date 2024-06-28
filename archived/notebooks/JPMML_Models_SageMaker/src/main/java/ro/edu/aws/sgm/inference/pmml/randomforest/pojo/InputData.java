package ro.edu.aws.sgm.inference.pmml.randomforest.pojo;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
 
public class InputData {
    
    @JsonProperty("data")
    private List<Features> featureList;

    @JsonCreator
    public InputData(List<Features> featureList) {
        super();
        this.featureList = featureList;
    }

    public List<Features> getFeatureList() {
        return featureList;
    }

    public void setFeatureList(List<Features> featureList) {
        this.featureList = featureList;
    }
    

}
