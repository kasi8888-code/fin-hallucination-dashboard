library(shiny)
library(bslib)
library(httr2)
library(jsonlite)
library(DT)
library(plotly)


# =========================================================
# API CONFIGURATION
# =========================================================

API_URL <- "http://127.0.0.1:8000"


# =========================================================
# API FUNCTION
# =========================================================

analyze_prompt <- function(prompt, reference = "") {

  body <- list(
    prompt = prompt,
    reference = if (nzchar(reference)) reference else NULL
  )

  request(paste0(API_URL, "/analyze")) |>
    req_method("POST") |>
    req_body_json(body) |>
    req_perform() |>
    resp_body_json(simplifyVector = TRUE)
}


# =========================================================
# SENTENCE SCORE CONVERTER
# =========================================================

get_sentence_df <- function(scores) {

  # No sentence scores
  if (is.null(scores) || length(scores) == 0) {

    return(
      data.frame(
        Sentence = character(),
        Risk = numeric(),
        Status = character(),
        stringsAsFactors = FALSE
      )
    )
  }


  # JSON returns sentence_scores as a list
  if (is.list(scores) && !is.data.frame(scores)) {

    sentences <- vapply(
      scores,
      function(x) {
        if (!is.null(x$sentence)) {
          as.character(x$sentence)
        } else {
          ""
        }
      },
      character(1)
    )


    risks <- vapply(
      scores,
      function(x) {

        if (!is.null(x$score)) {
          as.numeric(x$score)
        } else {
          NA_real_
        }

      },
      numeric(1)
    )


    hallucinated <- vapply(
      scores,
      function(x) {

        if (!is.null(x$is_hallucinated)) {
          as.numeric(x$is_hallucinated)
        } else {
          0
        }

      },
      numeric(1)
    )


  } else {

    sentences <- as.character(scores$sentence)

    risks <- as.numeric(scores$score)

    hallucinated <- as.numeric(
      scores$is_hallucinated
    )
  }


  data.frame(
    Sentence = sentences,
    Risk = risks,
    Status = ifelse(
      hallucinated == 1,
      "Potential Hallucination",
      "Low Risk"
    ),
    stringsAsFactors = FALSE
  )
}


# =========================================================
# USER INTERFACE
# =========================================================

ui <- page_navbar(

  title = "Financial Hallucination Risk Dashboard",


  # =======================================================
  # ANALYZE PAGE
  # =======================================================

  nav_panel(
    "Analyze",


    layout_sidebar(

      # ---------------------------------------------------
      # SIDEBAR
      # ---------------------------------------------------

      sidebar = sidebar(

        textAreaInput(
          inputId = "prompt",
          label = "Financial Question",
          placeholder = "Enter a financial question...",
          rows = 6
        ),


        textAreaInput(
          inputId = "reference",
          label = "Trusted Reference (Optional)",
          placeholder = "Enter trusted financial information...",
          rows = 6
        ),


        actionButton(
          inputId = "analyze",
          label = "Analyze Response",
          class = "btn-primary",
          width = "100%"
        )
      ),


      # ===================================================
      # MAIN CONTENT
      # ===================================================

      h2(
        "Analysis Result",
        style = "margin-bottom: 20px;"
      ),


      # ===================================================
      # METRIC CARDS
      # ===================================================

      layout_columns(

        value_box(
          title = "Risk Score",
          value = textOutput("risk_score")
        ),


        value_box(
          title = "Decision",
          value = textOutput("decision")
        ),


        value_box(
          title = "Model",
          value = textOutput("model")
        ),

        col_widths = c(4, 4, 4)
      ),


      # ===================================================
      # GENERATED RESPONSE
      # ===================================================

      card(

        card_header(
          "Generated Response"
        ),


        div(
          class = "response-box",

          uiOutput(
            "response"
          )
        )
      ),


      # ===================================================
      # SENTENCE LEVEL RISK
      # ===================================================

      card(

        card_header(
          "Sentence-Level Risk"
        ),


        div(
          class = "risk-chart-box",

          plotlyOutput(
            "risk_plot",
            height = "220px"
          )
        )
      ),


      # ===================================================
      # SENTENCE ANALYSIS TABLE
      # ===================================================

      card(

        card_header(
          "Sentence Analysis"
        ),


        div(
          class = "sentence-table-box",

          DTOutput(
            "sentence_table"
          )
        )
      )
    )
  ),


  # =======================================================
  # CUSTOM CSS
  # =======================================================

  tags$head(

    tags$style(
      HTML("

        /* ================================================
           GENERATED RESPONSE
           ================================================ */

        .response-box {

          min-height: 220px;

          max-height: 400px;

          overflow-y: auto;

          padding: 20px;

          font-size: 17px;

          line-height: 1.7;

          white-space: pre-wrap;

          background: #ffffff;

          border-radius: 6px;
        }


        /* ================================================
           SENTENCE RISK CHART
           ================================================ */

        .risk-chart-box {

          height: 220px;

          padding: 0px 10px 5px 10px;

          overflow: hidden;
        }


        /* ================================================
           SENTENCE TABLE
           ================================================ */

        .sentence-table-box {

          max-height: 280px;

          overflow-y: auto;

          padding: 5px;
        }


        /* ================================================
           CARD SPACING
           ================================================ */

        .bslib-card {

          margin-bottom: 16px;
        }


        /* ================================================
           CARD HEADERS
           ================================================ */

        .card-header {

          font-weight: 600;

          font-size: 16px;
        }


        /* ================================================
           METRIC VALUE BOXES
           ================================================ */

        .bslib-value-box {

          min-height: 120px;
        }


        /* ================================================
           SIDEBAR BUTTON
           ================================================ */

        #analyze {

          font-size: 16px;

          font-weight: 600;

          padding: 12px;
        }

      ")
    )
  )
)


# =========================================================
# SERVER
# =========================================================

server <- function(input, output, session) {


  # =======================================================
  # RUN ANALYSIS
  # =======================================================

  result <- eventReactive(
    input$analyze,

    {

      req(
        nzchar(trimws(input$prompt))
      )


      tryCatch(

        {

          analyze_prompt(

            prompt = input$prompt,

            reference = input$reference
          )
        },


        error = function(e) {

          showNotification(

            paste(
              "API Error:",
              e$message
            ),

            type = "error",

            duration = 8
          )


          NULL
        }
      )
    }
  )


  # =======================================================
  # RISK SCORE
  # =======================================================

  output$risk_score <- renderText({

    data <- result()

    req(data)


    sprintf(
      "%.3f",
      as.numeric(data$overall_score)
    )
  })


  # =======================================================
  # DECISION
  # =======================================================

  output$decision <- renderText({

    data <- result()

    req(data)


    if (!is.null(data$decision)) {

      data$decision

    } else {

      "N/A"
    }
  })


  # =======================================================
  # MODEL
  # =======================================================

  output$model <- renderText({

    data <- result()

    req(data)


    if (!is.null(data$model)) {

      data$model

    } else {

      "N/A"
    }
  })


  # =======================================================
  # GENERATED RESPONSE
  # =======================================================

  output$response <- renderUI({

    data <- result()

    req(data)


    div(
      data$response
    )
  })


  # =======================================================
  # SENTENCE LEVEL RISK GRAPH
  # =======================================================

  output$risk_plot <- renderPlotly({

    data <- result()

    req(data)


    df <- get_sentence_df(
      data$sentence_scores
    )


    # No data
    if (nrow(df) == 0) {

      return(
        plot_ly() |>
          layout(
            xaxis = list(
              visible = FALSE
            ),
            yaxis = list(
              visible = FALSE
            ),
            annotations = list(
              list(
                text = "No sentence-level data available",
                showarrow = FALSE
              )
            )
          )
      )
    }


    # Sentence numbers
    sentence_numbers <- seq_len(
      nrow(df)
    )


    plot_ly(

      df,

      x = ~sentence_numbers,

      y = ~Risk,

      type = "bar",

      text = ~paste(

        "<b>Sentence:</b>",
        Sentence,

        "<br><b>Risk:</b>",
        round(Risk, 3),

        "<br><b>Status:</b>",
        Status
      ),

      hoverinfo = "text"
    ) |>


      layout(

        xaxis = list(

          title = "Sentence",

          tickmode = "array",

          tickvals = sentence_numbers,

          ticktext = paste(
            "S",
            sentence_numbers
          )
        ),


        yaxis = list(

          title = "Hallucination Risk",

          range = c(0, 1)
        ),


        margin = list(

          l = 55,

          r = 20,

          t = 10,

          b = 45
        )
      )
  })


  # =======================================================
  # SENTENCE ANALYSIS TABLE
  # =======================================================

  output$sentence_table <- renderDT({

    data <- result()

    req(data)


    df <- get_sentence_df(
      data$sentence_scores
    )


    datatable(

      df,

      rownames = FALSE,


      colnames = c(

        "Sentence",

        "Risk Score",

        "Status"
      ),


      options = list(

        pageLength = 5,

        lengthChange = FALSE,

        searching = FALSE,

        ordering = TRUE,

        scrollX = TRUE
      )
    )
  })
}


# =========================================================
# START APPLICATION
# =========================================================

shinyApp(
  ui = ui,
  server = server
)