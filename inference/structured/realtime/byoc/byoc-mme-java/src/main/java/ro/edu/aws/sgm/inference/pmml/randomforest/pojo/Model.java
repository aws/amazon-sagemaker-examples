package ro.edu.aws.sgm.inference.pmml.randomforest.pojo;



public class Model{
    private String model_name;
    private String url;
    public Model(String model_name, String url) {
        this.model_name = model_name;
        this.url = url;
    }
    public String getModel_name() {
        return model_name;
    }
    public void setModel_name(String model_name) {
        this.model_name = model_name;
    }
    public String getUrl() {
        return url;
    }
    public void setUrl(String url) {
        this.url = url;
    }
}