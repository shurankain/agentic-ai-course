# Semantic Kernel: AI Framework by Microsoft

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Frameworks
**Previous:** [[02_Spring_AI|Spring AI]]
**Next:** [[04_JAX_Ecosystem|JAX Ecosystem]]

---

## Introduction to Semantic Kernel

Semantic Kernel is an SDK from Microsoft for integrating language models into applications. The project positions itself as a lightweight and extensible framework that provides primitives for building AI systems of any complexity.

The name "Semantic Kernel" reflects the central idea: a kernel that unifies the semantic capabilities of AI with traditional software components. The model is not an isolated service but an integral part of the application, capable of invoking functions, working with memory, and orchestrating complex workflows.

Microsoft develops Semantic Kernel as part of its AI strategy. The framework is deeply integrated with Azure AI Services but also works with other providers. .NET, Python, and Java are supported, making it accessible to a wide range of developers.

## Architectural Concepts

### Kernel as the System Core

Kernel is the central object of the framework, coordinating all AI operations. It holds service registrations, plugins, and functions. All interactions with the model go through the Kernel.

The Kernel is configured at creation time: model providers are specified, plugins are registered, and services are configured. Once created, the Kernel can be used to execute prompts, invoke functions, and orchestrate plans.

This centralized approach simplifies dependency and configuration management. The Kernel becomes the single entry point for all AI operations in the application.

### Plugins and Functions

Plugins in Semantic Kernel are groups of related functions. A function can be semantic (executed via an LLM) or native (regular code).

Semantic functions are defined through prompt templates. The developer describes what the LLM should do, and the Kernel executes the model call. Native functions are regular methods available to the model for invocation.

This separation reflects the reality of AI development: part of the logic is executed by the model (language understanding, text generation), and part by traditional code (computations, integrations, business logic).

### Memory and Context

Semantic Kernel provides abstractions for working with memory. Semantic Memory stores and retrieves information by semantic similarity. Chat History preserves the conversation history for multi-turn interactions.

Memory integrates with various backends: Azure AI Search, Pinecone, Qdrant, and in-memory implementations. A unified interface allows abstraction from the specific storage backend.

Context combines everything needed to execute a function: variables, history, and results of previous calls. It is passed between functions, ensuring execution continuity.

### Planners

One of the key capabilities of Semantic Kernel is automatic planning. The Planner analyzes the user's goal and composes a plan to achieve it using available functions.

**Connection to classical AI planning:**

The idea of automatic planning is not new — it dates back to the 1970s. STRIPS (1971) modeled the world as a set of predicates like on(A,B) for blocks, defined operators with preconditions and effects, and performed planning through state-space search. PDDL (1998) standardized the domain description language, separating domain (what is possible) and problem (the specific goal), using planners such as GraphPlan and FastDownward.

Semantic Kernel adapts these ideas for LLMs. Instead of explicitly modeling the world through predicates, the "world" is understood implicitly by the LLM. Operators are functions with textual descriptions, and planning occurs through LLM reasoning and prompt engineering. The advantage is the ability to work with natural language goals without formal modeling. The disadvantage is a less formal approach, making it harder to guarantee correctness.

**Types of planners:**

Sequential Planner creates the entire plan upfront as a linear sequence of steps — suitable for predictable tasks. Stepwise Planner works iteratively, deciding at each step what to do next with reflection — better for complex tasks with uncertainty. Handlebars Planner generates a plan as a template — useful for repeating patterns. Function Calling Planner uses native LLM planning capabilities — optimal for simple tool-heavy tasks.

Planning transforms a set of functions into an intelligent system capable of independently determining how to achieve a goal.

## Plugin Development

### Native Plugins

Native plugins are classes with methods marked by attributes. Each public method with a KernelFunction attribute becomes a function available to the Kernel.

Method and parameter descriptions are critically important. The model decides whether to invoke a function based on these descriptions. Clear descriptions increase the likelihood of correct usage.

Native plugins can perform any operations: HTTP requests, database operations, computations, integrations with external systems. They extend the model's capabilities, giving it "hands" to act in the real world.

### Semantic Plugins

Semantic functions are defined through prompt templates. The template describes what the LLM should do, with placeholders for parameters.

Templates are stored in separate files, simplifying editing without changing code. Function configuration (model parameters, description) is stored alongside the template.

This separation of prompts and code is a good practice. Prompts can be versioned, tested, and optimized independently of the application logic.

### Function Composition

Functions can be combined to create complex workflows. The result of one function is passed as input to another. The Kernel coordinates execution, managing the context.

Composition can be explicit (the developer defines the sequence) or automatic (the planner composes the plan). Both approaches use the same primitives.

## Provider Integration

### Azure OpenAI

Semantic Kernel is deeply integrated with Azure OpenAI Service. Configuration requires an endpoint, deployment name, and key. All Azure models are supported: GPT-4, GPT-3.5, and embedding models.

Azure meets enterprise requirements: compliance, regionalization, and private endpoints. For enterprise applications, Azure is often the preferred choice.

### OpenAI

Direct integration with the OpenAI API is also supported. Specifying an API key and model is sufficient. This is convenient for development and small projects.

### Hugging Face and Local Models

Semantic Kernel supports Hugging Face models and local solutions. This opens up the ability to work without cloud APIs, which is important for privacy-sensitive scenarios.

Integration with Ollama allows using local open-source models without changing the application code.

## Connectors and Extensions

### Extensibility Through Connectors

Semantic Kernel uses a connector system for integration with external services. AI Connector abstracts model providers. Memory Connector abstracts memory stores.

Creating a custom connector allows integrating any service. Implementing the corresponding interface is all that is required.

### Built-in Integrations

The framework includes integrations with popular services: Azure Cognitive Services, Microsoft Graph, and various vector databases. This covers typical enterprise scenarios without additional development.

### Agents and Copilots

Semantic Kernel provides abstractions for building agents and copilots. An Agent is an entity capable of performing tasks autonomously using plugins and planning.

The Copilot pattern involves user interaction where AI assists but control remains with the human. Semantic Kernel supports both patterns through configurable agent behavior.

## Observability and Debugging

### Telemetry

Semantic Kernel integrates with OpenTelemetry for tracing and metrics. Every function call, model request, and plan execution generates telemetry.

This is critically important for production systems. Understanding which functions are called, how many tokens are used, and where delays occur is the foundation of optimization and debugging.

### Logging

Structured logging provides execution details. Prompts, responses, plans, and errors are all logged with the appropriate level of detail.

Integration with standard logging frameworks simplifies correlating AI logs with the rest of the application.

## Practical Architecture

The basic architecture of a Semantic Kernel application includes several key patterns.

**Kernel Creation and Initialization:** The Kernel is created through a builder pattern specifying the AI service provider. Basic initialization includes connecting to OpenAI or Azure OpenAI with a model (e.g., gpt-4o) and an API key from environment variables. Once the service is created, it is registered in the Kernel. The ready Kernel is used to execute prompts asynchronously.

**Working with Prompt Templates:** Semantic Kernel supports parameterized templates with placeholders. Variables are passed through a KernelFunctionArguments object. This allows reusing prompts with different input data without modifying their text.

**Creating Native Plugins:** Native plugins are implemented as regular classes. Methods that should be available to the model are annotated with @DefineKernelFunction specifying the function name and description. Method parameters are annotated with @KernelFunctionParameter with a detailed description of their purpose. These descriptions are critically important — the model uses them to decide whether to invoke the function.

Plugins can contain any logic: getting the current time, performing mathematical operations, working with in-memory data, HTTP requests, and database integrations. After creating the plugin class, it is registered in the Kernel. Once registered, the model can independently invoke these functions when processing user requests.

**Semantic Functions:** Semantic functions are created programmatically. They include a prompt template with instructions for the model and variable placeholders, execution settings for parameters like temperature, max tokens, and top-p, as well as the function name and description.

Examples of typical semantic functions: text summarization with a specified number of sentences, sentiment analysis limited to a single word response (POSITIVE/NEGATIVE/NEUTRAL), and extracting key topics from text. Functions are invoked asynchronously with arguments.

**Working with Semantic Memory:** Semantic Memory provides semantic information retrieval. Using it requires creating an embedding service for text vectorization, choosing a store (InMemoryMemoryStore for development or production solutions like Qdrant or Pinecone), creating a SemanticTextMemory combining the store and embedding service, creating a collection for document organization, and adding documents with text, ID, and metadata.

Search is performed by specifying a query, the number of results, and a relevance threshold from 0.0 to 1.0. Results are ranked by semantic similarity.

Integration with the Kernel for the RAG pattern: before sending a question to the model, a semantic search for relevant information is performed, found documents are added to the prompt as context, and the model answers based on the provided context. This significantly reduces hallucinations and allows the model to work with up-to-date domain-specific information.

**Automatic Planning:** The StepwisePlanner automates achieving complex goals. It analyzes a natural language goal, decomposes it into subtasks, and determines the sequence of function calls. Planner configuration includes the maximum number of iterations and the minimum time between steps.

When creating the planner, the Kernel with registered plugins is passed to it. The planner analyzes available functions and their descriptions, then iteratively executes steps: evaluating the current state, selecting the next action, invoking a function, analyzing the result, and adjusting the plan as needed. This enables solving tasks that require a sequence of actions without explicitly programming the logic.

## Key Takeaways

Semantic Kernel is a powerful and flexible framework for building AI applications. The central Kernel abstraction unifies models, functions, and memory. Plugins extend the model's capabilities with native code and semantic functions.

Planners automate goal achievement, transforming a set of functions into an intelligent system. Deep integration with Azure makes Semantic Kernel a natural choice for the Microsoft ecosystem.

Support for multiple programming languages and model providers ensures flexibility. Observability through OpenTelemetry prepares the system for production use.

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Frameworks
**Previous:** [[02_Spring_AI|Spring AI]]
**Next:** [[04_JAX_Ecosystem|JAX Ecosystem]]

## Practical Code Examples

### Kernel Creation and Configuration

```java
import com.microsoft.semantickernel.Kernel;
import com.microsoft.semantickernel.orchestration.InvocationContext;
import com.microsoft.semantickernel.services.chatcompletion.ChatCompletionService;
import com.azure.ai.openai.OpenAIAsyncClient;
import com.azure.ai.openai.OpenAIClientBuilder;

// Create Azure OpenAI client
OpenAIAsyncClient client = new OpenAIClientBuilder()
    .credential(new AzureKeyCredential(System.getenv("AZURE_OPENAI_KEY")))
    .endpoint(System.getenv("AZURE_OPENAI_ENDPOINT"))
    .buildAsyncClient();

// Create chat completion service
ChatCompletionService chatService = ChatCompletionService.builder()
    .withOpenAIAsyncClient(client)
    .withModelId("gpt-4o")
    .build();

// Build Kernel with service registration
Kernel kernel = Kernel.builder()
    .withAIService(ChatCompletionService.class, chatService)
    .build();
```

### Defining a Native Plugin

```java
import com.microsoft.semantickernel.semanticfunctions.annotations.DefineKernelFunction;
import com.microsoft.semantickernel.semanticfunctions.annotations.KernelFunctionParameter;

public class TimePlugin {

    /**
     * Gets the current time in the specified format
     */
    @DefineKernelFunction(
        name = "getCurrentTime",
        description = "Returns the current time in the specified time zone"
    )
    public String getCurrentTime(
        @KernelFunctionParameter(
            name = "timezone",
            description = "Time zone in the format 'Europe/Kiev', 'America/New_York'"
        ) String timezone
    ) {
        ZoneId zone = ZoneId.of(timezone);
        ZonedDateTime time = ZonedDateTime.now(zone);
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("HH:mm:ss");
        return time.format(formatter);
    }

    /**
     * Calculates the difference between two dates
     */
    @DefineKernelFunction(
        name = "calculateDateDifference",
        description = "Calculates the number of days between two dates"
    )
    public long calculateDateDifference(
        @KernelFunctionParameter(
            name = "startDate",
            description = "Start date in ISO format 'yyyy-MM-dd'"
        ) String startDate,
        @KernelFunctionParameter(
            name = "endDate",
            description = "End date in ISO format 'yyyy-MM-dd'"
        ) String endDate
    ) {
        LocalDate start = LocalDate.parse(startDate);
        LocalDate end = LocalDate.parse(endDate);
        return ChronoUnit.DAYS.between(start, end);
    }
}
```

### Registering and Invoking Plugins

```java
// Import plugin into Kernel
TimePlugin timePlugin = new TimePlugin();
kernel.importPluginFromObject(timePlugin, "TimePlugin");

// Create function arguments
KernelFunctionArguments arguments = KernelFunctionArguments.builder()
    .withVariable("timezone", "Europe/Kiev")
    .build();

// Get function from Kernel
KernelFunction<?> timeFunction = kernel
    .getFunction("TimePlugin", "getCurrentTime");

// Synchronous function invocation
FunctionResult<Object> result = timeFunction.invoke(kernel, arguments, null);
System.out.println("Current time: " + result.getResult());

// Asynchronous function invocation
CompletableFuture<FunctionResult<Object>> asyncResult =
    timeFunction.invokeAsync(kernel, arguments, null);

asyncResult.thenAccept(res -> {
    System.out.println("Current time (async): " + res.getResult());
});
```

### Working with Prompts and Automatic Function Calling

```java
import com.microsoft.semantickernel.orchestration.PromptExecutionSettings;

// Register a plugin with business logic
public class WeatherPlugin {

    @DefineKernelFunction(
        name = "getWeather",
        description = "Gets the current weather for a specified city"
    )
    public String getWeather(
        @KernelFunctionParameter(
            name = "city",
            description = "City name"
        ) String city
    ) {
        // Simulating a weather API call
        return String.format("In %s it is currently +15°C, cloudy", city);
    }
}

// Register the plugin
WeatherPlugin weatherPlugin = new WeatherPlugin();
kernel.importPluginFromObject(weatherPlugin, "WeatherPlugin");

// Configure automatic function calling
PromptExecutionSettings settings = PromptExecutionSettings.builder()
    .withTemperature(0.7)
    .withMaxTokens(500)
    .withFunctionCallBehavior(
        FunctionCallBehavior.allowAllFunctions(true)
    )
    .build();

// Execute prompt with automatic function calling
String userPrompt = "What is the current weather in Kyiv?";

InvocationContext context = InvocationContext.builder()
    .withPromptExecutionSettings(settings)
    .build();

// The model will automatically determine the need to call getWeather
FunctionResult<Object> answer = kernel.invokePromptAsync(userPrompt)
    .withInvocationContext(context)
    .block();

System.out.println("Answer: " + answer.getResult());
// Output: "Answer: In Kyiv it is currently +15°C, cloudy"
```

These examples demonstrate the key capabilities of the Semantic Kernel Java SDK: kernel creation and configuration, defining reusable plugins with detailed function descriptions, and explicit and automatic function invocation through the model. The `@DefineKernelFunction` and `@KernelFunctionParameter` annotations provide the metadata required for function calling.
