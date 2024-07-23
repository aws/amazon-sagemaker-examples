package ro.edu.aws.sgm.inference.pmml.randomforest.app;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.autoconfigure.EnableAutoConfiguration;
import org.springframework.context.annotation.ComponentScan;


@SpringBootConfiguration
@EnableAutoConfiguration
@ComponentScan(basePackages = "ro.edu.aws.sgm.inference.pmml.randomforest")

public class SGMLauncher {
  public static void main(String[] args) {
      serve();
  }

  public static void serve() {
    SpringApplication.run(SGMLauncher.class);
  }
}
