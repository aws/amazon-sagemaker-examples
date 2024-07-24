library(shiny)
library(caret)
library(gbm)
library(e1071)

source('breast_cancer_modeling.r')
test_data <- readRDS('./breast_cancer_test_data.rds')
gbmFit <- readRDS('./gbm_model.rds')
preProcessor <- readRDS('./preProcessor.rds')
test_data_transformed <- predict(preProcessor, test_data)
prediction <- predict(gbmFit, newdata = test_data_transformed[,2:10], type = "prob")

inputs1 <- c("Clump Thickness" = "Cl.thickness",
             "Uniformity of Cell Size" = "Cell.size",
             "Uniformity of Cell Shape" = "Cell.shape",
             "Marginal Adhesion" = "Marg.adhesion",
             "Single Epithelial Cell Size" = "Epith.c.size",
             "Bare Nuclei" = "Bare.nuclei",
             "Bland Chromatin" = "Bl.cromatin",
             "Normal Nucleoli" = "Normal.nucleoli",
             "Mitoses" = "Mitoses")

inputs2 <- c("Uniformity of Cell Size" = "Cell.size",
             "Clump Thickness" = "Cl.thickness",
             "Uniformity of Cell Shape" = "Cell.shape",
             "Marginal Adhesion" = "Marg.adhesion",
             "Single Epithelial Cell Size" = "Epith.c.size",
             "Bare Nuclei" = "Bare.nuclei",
             "Bland Chromatin" = "Bl.cromatin",
             "Normal Nucleoli" = "Normal.nucleoli",
             "Mitoses" = "Mitoses")


# Define UI for the app ----
ui <- fluidPage(
  
  # App title ----
  titlePanel("Breast Cancer"),
  
  # Sidebar layout with input and output definitions ----
  sidebarLayout(
    
    # Sidebar panel for inputs ----
    sidebarPanel(
      # Input: Decimal interval with step value ----
      sliderInput("threshold", "Probability Threshold:",
                  min = 0, max = 1,
                  value = 0.5, step = 0.01),
      
      # Input: Selector for variable to plot on x axis ----
      selectInput("variable_x", "Variable on X:",
                  inputs1),
      
      # Input: Selector for variable to plot on y axis ----
      selectInput("variable_y", "Variable on Y:",
                  inputs2),
    ),
    
    # Main panel for displaying outputs ----
    mainPanel(
      
      # Output: Formatted text for caption ----
      h3(textOutput("caption")),
      
      # Output: prediction outcome
      tableOutput("predictions"),
      
      # Output: Verbatim text for data summary ----
      verbatimTextOutput("summary"),
      
      # Output: Formatted text for formula ----
      h3(textOutput("formula")),
      
      # Output: Plot of the data ----
      # was  click = "plot_click"
      plotOutput("scatterPlot", brush = "plot_brush"),
      
      # Output: present click info
      tableOutput("info")
      
    )
  )
)

# Define server logic to plot various variables ----
server <- function(input, output) {
  
  # Compute the formula text ----
  # This is in a reactive expression since it is shared by the
  # output$caption function
  formulaText <- reactive({
    paste(input$variable_y, "~", input$variable_x)
  })
  
  # Compute the formula text ----
  # This is in a reactive expression since it is shared by the
  # output$caption function
  total_count <- reactive({
    data.frame(Class = colnames(prediction),
               Count = c(sum(prediction$malignant<input$threshold),
                         sum(prediction$malignant>=input$threshold)))
  })
  
  # Compute the formula text ----
  # This is in a reactive expression
  threshold_proba <- reactive({
    cbind(Prediction = ifelse(prediction$malignant>=input$threshold, 
                              "malignant", "benign"),
          test_data)
  })
  
  # return prediction summary
  output$predictions <- renderTable({
    total_count()
  })
  
  # Return the formula text for printing as a caption ----
  output$caption <- renderText({
    "Breast cancer test data summary"
  })
  
  # Generate a summary of the dataset ----
  # The output$summary depends on the datasetInput reactive
  # expression, so will be re-executed whenever datasetInput is
  # invalidated, i.e. whenever the input$dataset changes
  output$summary <- renderPrint({
    summary(test_data)
  })
  
  # Return the formula text for printing as a caption ----
  output$formula <- renderText({
    formulaText()
  })
  
  # Generate a plot of the requested variables ----
  # and only exclude outliers if requested
  output$scatterPlot <- renderPlot({
    plot(as.formula(formulaText()), data = threshold_proba())
  })
  
  output$info <- renderTable({
    brushedPoints(threshold_proba(), input$plot_brush, 
                  xvar = input$variable_x, yvar = input$variable_y)
  })
  
}

# Create Shiny app ----
shinyApp(ui, server)