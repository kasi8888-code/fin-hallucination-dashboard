# =============================================================================
# Financial LLM Hallucination Detection - R Shiny Dashboard
# -----------------------------------------------------------------------------
# Front-end ("Builder" role) that consumes the FastAPI hallucination-risk
# backend. Endpoints used:
#   GET  /              -> health check
#   POST /analyze       -> one-shot verdict for a prompt (+ optional reference)
#   GET  /history       -> aggregate table of past analyses
#   GET  /analysis/{id} -> full detail incl. the N sampled LLM responses
#
# Run:  Rscript run.R        (or open app.R in RStudio and press Run App)
# =============================================================================

suppressPackageStartupMessages({
  library(shiny)
  library(shinydashboard)
  library(httr)
  library(jsonlite)
  library(DT)
  library(plotly)
  library(dplyr)
  library(ggplot2)
})

`%||%` <- function(a, b) if (is.null(a) || length(a) == 0) b else a

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Override via Sys.setenv(HALLUC_API_URL = "http://localhost:8000")
API_BASE <- Sys.getenv("HALLUC_API_URL", unset = "http://127.0.0.1:8000")

# Decision colour scheme shared by UI cards, tables and plots.
DECISION_COLORS <- c(
  "Faithful"     = "#1e8449",   # green
  "Suspicious"   = "#e67e22",   # orange
  "Hallucinated" = "#c0392b"    # red
)

# ---------------------------------------------------------------------------
# Backend API client
# ---------------------------------------------------------------------------

api_get_text <- function(path, timeout_secs = 10) {
  url <- paste0(API_BASE, path)
  resp <- GET(url, timeout(seconds = timeout_secs))
  if (status_code(resp) >= 400) {
    stop("Backend returned HTTP ", status_code(resp), " for ", url)
  }
  content(resp, "text", encoding = "UTF-8")
}

api_post_analyze <- function(prompt, reference = NULL, timeout_secs = 180) {
  body <- list(prompt = prompt)
  if (!is.null(reference) && nzchar(trimws(reference))) {
    body$reference <- reference
  }
  resp <- POST(
    paste0(API_BASE, "/analyze"),
    encode = "json",
    body = body,
    timeout(seconds = timeout_secs)
  )
  if (status_code(resp) >= 400) {
    stop("Backend error (HTTP ", status_code(resp), "): ",
         substr(content(resp, "text", encoding = "UTF-8"), 1, 500))
  }
  fromJSON(content(resp, "text", encoding = "UTF-8"), simplifyVector = FALSE)
}

api_health <- function(timeout_secs = 3) {
  ok <- tryCatch({
    resp <- GET(paste0(API_BASE, "/"), timeout(seconds = timeout_secs))
    status_code(resp) == 200
  }, error = function(e) FALSE)
  ok
}

api_history <- function() {
  fromJSON(api_get_text("/history"), simplifyVector = TRUE)
}

api_detail <- function(analysis_id) {
  fromJSON(api_get_text(paste0("/analysis/", analysis_id)), simplifyVector = FALSE)
}

# ---------------------------------------------------------------------------
# Small display helpers
# ---------------------------------------------------------------------------

decision_color <- function(decision) {
  color <- unname(DECISION_COLORS[decision])
  if (is.na(color)) "#7f8c8d" else color
}

decision_box_color <- function(decision) {
  switch(decision,
         Faithful     = "green",
         Suspicious   = "yellow",
         Hallucinated = "red",
         "light-blue")
}

risk_label <- function(score) paste0(round(as.numeric(score) * 100, 1), "%")

# Sentence-level risk list-of-lists (from the API) -> data.frame.
sentence_frame <- function(sentence_scores) {
  if (is.null(sentence_scores) || length(sentence_scores) == 0) {
    return(NULL)
  }
  df <- do.call(rbind, lapply(sentence_scores, function(s) {
    data.frame(
      Sentence = s$sentence,
      Score    = as.numeric(s$score),
      Flag     = ifelse(as.integer(s$is_hallucinated) == 1, "FLAGGED", "ok"),
      stringsAsFactors = FALSE
    )
  }))
  df$Idx <- seq_len(nrow(df))
  df
}

# Build a DT table of sentence-level risks (reused in both tabs).
sentence_datatable <- function(df) {
  if (is.null(df) || nrow(df) == 0) {
    return(datatable(
      data.frame(Message = "No sentence-level scores were returned."),
      rownames = FALSE, options = list(dom = "t")
    ))
  }
  dat <- data.frame(
    `#`        = df$Idx,
    Sentence   = df$Sentence,
    `Risk`     = round(df$Score, 3),
    Status     = df$Flag,
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  datatable(
    dat,
    rownames = FALSE,
    escape   = FALSE,
    options = list(
      pageLength = 8,
      lengthMenu = c(5, 8, 15),
      columnDefs = list(
        list(width = "30px", targets = 0),
        list(width = "90px", targets = 2)
      ),
      dom = "fltip"
    )
  ) %>%
    formatStyle(
      "Risk",
      background = styleColorBar(c(0, 1), "#e8b06b"),
      color = styleInterval(c(0.52, 0.60), c("#111", "#111", "#fff")),
      fontWeight = styleInterval(0.6, c("normal", "bold"))
    ) %>%
    formatStyle(
      "Sentence",
      fontSize = "13px",
      whiteSpace = "pre-wrap"
    ) %>%
    formatStyle(
      "Status",
      target = "row",
      backgroundColor = styleEqual("FLAGGED", "rgba(192,57,43,.15)")
    )
}

# Horizontal bars for sentence-level risk (reused in both tabs).
sentence_plotly <- function(df, title) {
  if (is.null(df) || nrow(df) == 0) return(NULL)
  p <- ggplot(df, aes(x = reorder(paste0("#", Idx), Idx),
                      y = Score, fill = Score,
                      text = paste0("Sentence: ", Sentence,
                                    "<br>Risk: ", round(Score, 3)))) +
    geom_col(width = 0.7) +
    scale_fill_gradient2(low = "#1e8449", mid = "#e67e22",
                         high = "#c0392b", midpoint = 0.5,
                         limits = c(0, 1)) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(x = NULL, y = "Hallucination risk", title = title) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none",
          axis.text.x = element_text(angle = 45, hjust = 1))
  ggplotly(p, tooltip = "text") %>%
    layout(margin = list(b = 60))
}

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

ui <- dashboardPage(
  skin = "black",

  dashboardHeader(
    title = tagList(
      span("LLM Hallucination Risk", style = "font-size:18px"),
      span(" - Financial QA", style = "font-size:12px;opacity:.75")
    )
  ),

  dashboardSidebar(
    width = 220,
    sidebarMenu(
      id = "tabs",
      menuItem("Analyze", tabName = "analyze", icon = icon("flask")),
      menuItem("History", tabName = "history", icon = icon("table"))
    )
  ),

  dashboardBody(
    tags$head(tags$style(HTML("
      .small-box .icon { font-size: 55px; }
      pre.llm-out { white-space: pre-wrap; word-break: break-word;
                    max-height: 260px; overflow-y: auto;
                    background: #f7f7f7; padding: 10px; }
    "))),

    tabItems(
      # ================================================================
      # Tab 1: Analyze a prompt
      # ================================================================
      tabItem(
        tabName = "analyze",
        fluidRow(
          box(
            title = "Backend status", width = 12, collapsible = TRUE,
            collapsed = TRUE, status = "primary", solidHeader = TRUE,
            textOutput("api_status")
          )
        ),
        fluidRow(
          box(
            title = "New analysis", width = 12, status = "primary",
            solidHeader = TRUE,
            textAreaInput(
              "prompt",
              label = "Financial question / prompt",
              value = "What was Apple's reported revenue for Q3 FY2024?",
              rows = 3, width = "100%"
            ),
            textAreaInput(
              "reference",
              label = p("Trusted reference (optional - strongly recommended, it grounds the answer)",
                        span("If empty, only self-consistency is used",
                             style = "color:#888;font-weight:normal")),
              value = paste(
                "Apple reported revenue of $85.8 billion for its fiscal",
                "Q3 2024, with net income of $21.4 billion and diluted EPS",
                "of $1.40."
              ),
              rows = 3, width = "100%"
            ),
            fluidRow(
              column(6,
                actionButton("btn_analyze", "Run hallucination check",
                             icon = icon("play"), class = "btn-primary",
                             width = "100%")
              ),
              column(6,
                actionButton("btn_seed_hi", "Fill hallucinated example",
                             icon = icon("dice"), width = "100%")
              )
            )
          )
        ),
        uiOutput("analyze_result")
      ),

      # ================================================================
      # Tab 2: History
      # ================================================================
      tabItem(
        tabName = "history",
        fluidRow(
          box(
            title = "Past analyses", width = 12, status = "primary",
            solidHeader = TRUE,
            div(
              style = "margin-bottom:6px",
              actionButton("btn_refresh_history", "Refresh", icon = icon("sync"))
            ),
            DTOutput("history_table")
          )
        ),
        uiOutput("history_detail_ui"),
        uiOutput("history_sent_ui")
      )
    )
  )
)

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

server <- function(input, output, session) {

  rv <- reactiveValues(
    api_ok        = NULL,
    last_result   = NULL,     # parsed POST /analyze payload
    detail        = NULL,     # sampled responses of the last /analyze
    history       = NULL,
    selected_id   = NULL,     # analysis id selected in the History tab
    detail_result = NULL      # payload of GET /analysis/{id} (history)
  )

  # ---- Backend health banner ---------------------------------------------
  observe({
    rv$api_ok <- api_health()
    invalidateLater(15000, session)
  })

  output$api_status <- renderText({
    if (isTRUE(rv$api_ok)) {
      paste0("Connected to backend at ", API_BASE)
    } else {
      paste0("Backend unreachable at ", API_BASE,
             ". Start it first:  cd backend ; uvicorn main:app --reload")
    }
  })

  outputOptions(output, "api_status", suspendWhenHidden = FALSE)

  # ---- Analyse -----------------------------------------------------------
  observeEvent(input$btn_analyze, {
    prompt <- trimws(input$prompt)
    if (!nzchar(prompt)) {
      showNotification("Please enter a prompt.", type = "error")
      return(NULL)
    }
    reference <- trimws(input$reference)
    reference <- if (nzchar(reference)) reference else NULL

    if (!isTRUE(rv$api_ok)) {
      showNotification("Backend is not reachable. Start uvicorn first.",
                       type = "error")
      return(NULL)
    }

    showNotification(
      "Sampling the LLM multiple times and scoring... this can take a minute.",
      type = "message", duration = NULL)
    result <- tryCatch(
      api_post_analyze(prompt, reference),
      error = function(e) {
        showNotification(paste("Analysis failed:", conditionMessage(e)),
                         type = "error")
        NULL
      }
    )
    if (is.null(result)) return(NULL)

    rv$last_result <- result
    rv$detail <- tryCatch(api_detail(result$id), error = function(e) NULL)
    rv$selected_id <- NULL
    updateTabItems(session, "tabs", "analyze")
  })

  observeEvent(input$btn_seed_hi, {
    updateTextAreaInput(
      session, "prompt",
      value = paste(
        "What was the exact year-over-year revenue growth of JPMorgan",
        "Chase in fiscal 2025, to two decimal places in percent?"))
    updateTextAreaInput(session, "reference", value = "")
  })

  # ---- Analyze: result panel --------------------------------------------
  output$analyze_result <- renderUI({
    res <- rv$last_result
    if (is.null(res)) return(NULL)

    decision <- res$decision %||% "Unknown"
    score    <- as.numeric(res$overall_score)

    tagList(
      fluidRow(
        valueBox(risk_label(score), "Fused hallucination risk",
                 icon = icon("tachometer-alt"), color = "light-blue", width = 4),
        valueBox(decision, "Model verdict",
                 icon = icon("gavel"),
                 color = decision_box_color(decision), width = 4),
        valueBox(res$model %||% "n/a", "LLM model",
                 icon = icon("robot"), color = "purple", width = 4)
      ),
      fluidRow(
        box(
          title = "Response (primary sample)", width = 12,
          status = "primary", collapsible = TRUE,
          pre(class = "llm-out", htmltools::htmlEscape(res$response))
        )
      ),
      fluidRow(
        box(
          title = "Sentence-level risk", width = 12, status = "warning",
          solidHeader = TRUE, collapsible = TRUE,
          plotlyOutput("sentence_plot"),
          br(),
          DTOutput("sentence_table")
        )
      ),
      fluidRow(
        box(
          title = "Sampled responses (SelfCheckGPT comparison)",
          width = 12, status = "info", solidHeader = TRUE,
          collapsible = TRUE, collapsed = TRUE,
          uiOutput("sampled_responses")
        )
      )
    )
  })

  output$sentence_plot <- renderPlotly({
    res <- rv$last_result
    if (is.null(res)) return(NULL)
    sentence_plotly(sentence_frame(res$sentence_scores),
                    "Per-sentence hallucination risk of the primary response")
  })

  output$sentence_table <- renderDT({
    res <- rv$last_result
    if (is.null(res)) return(NULL)
    sentence_datatable(sentence_frame(res$sentence_scores))
  })

  output$sampled_responses <- renderUI({
    det <- rv$detail
    if (is.null(det) || is.null(det$llm_responses)) {
      return(p("No sampled responses available."))
    }
    tagList(lapply(det$llm_responses, function(r) {
      wellPanel(
        style = "margin-bottom:6px",
        h5(strong(paste0("Sample #", r$response_number))),
        pre(class = "llm-out", htmltools::htmlEscape(r$response))
      )
    }))
  })

  # ---- History -----------------------------------------------------------
  load_history <- function() {
    rv$history <- tryCatch({
      h <- api_history()
      if (length(h) == 0 || nrow(h) == 0) NULL else h
    }, error = function(e) {
      showNotification(paste("Could not load history:", conditionMessage(e)),
                       type = "error")
      NULL
    })
  }

  observeEvent(input$tabs, {
    if (input$tabs == "history") load_history()
  })
  observeEvent(input$btn_refresh_history, load_history())

  output$history_table <- renderDT({
    h <- rv$history
    if (is.null(h)) {
      return(datatable(
        data.frame(Message = "No analyses yet - run one in the Analyze tab."),
        rownames = FALSE, options = list(dom = "t")
      ))
    }
    dat <- data.frame(
      ID       = as.integer(h$id),
      Created  = as.character(h$created_at),
      Prompt   = substr(h$prompt, 1, 90),
      Decision = h$decision %||% "n/a",
      Risk     = round(as.numeric(h$overall_score), 3),
      stringsAsFactors = FALSE
    )
    datatable(
      dat, rownames = FALSE, selection = "single",
      options = list(
        pageLength = 10,
        dom = "fltip",
        columnDefs = list(
          list(width = "50px", targets = 0),
          list(width = "150px", targets = 1)
        )
      )
    ) %>%
      formatStyle("Decision",
                  color = styleEqual(names(DECISION_COLORS),
                                     unname(DECISION_COLORS))) %>%
      formatStyle("Risk", background = styleColorBar(c(0, 1), "#e8b06b"))
  })

  # Clicking a row (and DT single-selection) both drive the detail panel.
  observeEvent(input$history_table_rows_selected, {
    h <- rv$history
    if (is.null(h) || length(input$history_table_rows_selected) == 0) return(NULL)
    idx <- input$history_table_rows_selected[1]
    id  <- as.integer(h$id[idx])
    rv$selected_id <- id
    rv$detail_result <- tryCatch(api_detail(id), error = function(e) NULL)
  })

  # Detail header + samples for the selected history row (pure markup).
  output$history_detail_ui <- renderUI({
    det <- rv$detail_result
    if (is.null(det)) return(NULL)
    decision <- det$decision %||% "Unknown"
    score    <- as.numeric(det$overall_score)

    tagList(
      fluidRow(
        valueBox(risk_label(score), paste0("Analysis #", det$id, " - risk"),
                 icon = icon("tachometer-alt"), color = "light-blue", width = 4),
        valueBox(decision, "Model verdict", icon = icon("gavel"),
                 color = decision_box_color(decision), width = 4),
        valueBox(det$model %||% "n/a", "LLM model",
                 icon = icon("robot"), color = "purple", width = 4)
      ),
      fluidRow(
        box(title = "Prompt", width = 6, status = "info",
            pre(class = "llm-out", htmltools::htmlEscape(det$prompt))),
        box(title = "Response", width = 6, status = "primary",
            pre(class = "llm-out", htmltools::htmlEscape(det$response)))
      ),
      fluidRow(
        box(
          title = "Sampled responses", width = 12, status = "info",
          solidHeader = TRUE, collapsible = TRUE,
          tagList(lapply(det$llm_responses, function(r) {
            wellPanel(
              style = "margin-bottom:6px",
              h5(strong(paste0("Sample #", r$response_number))),
              pre(class = "llm-out", htmltools::htmlEscape(r$response))
            )
          }))
        )
      )
    )
  })

  # Sentence-risk table/plot for the selected history row.
  output$history_sent_ui <- renderUI({
    det <- rv$detail_result
    if (is.null(det)) return(NULL)
    fluidRow(
      box(
        title = "Sentence-level risk", width = 12, status = "warning",
        solidHeader = TRUE, collapsible = TRUE,
        plotlyOutput("history_sentence_plot"),
        br(),
        DTOutput("history_sentence_table")
      )
    )
  })

  output$history_sentence_plot <- renderPlotly({
    det <- rv$detail_result
    if (is.null(det)) return(NULL)
    sentence_plotly(sentence_frame(det$sentence_scores),
                    "Per-sentence hallucination risk")
  })

  output$history_sentence_table <- renderDT({
    det <- rv$detail_result
    if (is.null(det)) return(NULL)
    sentence_datatable(sentence_frame(det$sentence_scores))
  })
}

shinyApp(ui = ui, server = server)
