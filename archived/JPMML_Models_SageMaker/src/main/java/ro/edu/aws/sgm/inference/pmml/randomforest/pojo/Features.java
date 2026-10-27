package ro.edu.aws.sgm.inference.pmml.randomforest.pojo;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;


public class Features {
    
    @JsonProperty("features")
    private List<String> features;

    @JsonCreator
    public Features(List<String> features) {
        super();
        this.features = features;
    }

    public List<String> getFeatures() {
        return features;
    }

    public void setFeatures(List<String> features) {
        this.features = features;
    }

}
