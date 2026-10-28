# Spring AI: The Official AI Framework of the Spring Ecosystem

Spring AI is the official Spring ecosystem project for integrating applications with artificial intelligence systems. This framework continues the Spring tradition of providing unified, productive abstractions for complex technological challenges.

**Spring AI 1.0 GA** was released in May 2025, marking the transition from experimental to production-ready. **Spring AI 1.1 GA** followed in November 2025 with expanded MCP support and additional model integrations. **Spring AI 2.0.0 GA** was released on **June 12, 2026**, a major version requiring **Spring Boot 4.0/4.1** and **Spring Framework 7.0**. Key 2.0 additions: **ToolSearchToolCallingAdvisor** for progressive tool disclosure (indexes all tools per session, exposes only a built-in `toolSearchTool` for on-demand retrieval: three ToolIndex implementations: vector store, Lucene, regex), enhanced observability for chat models, and improved OpenAI options builder pattern. Spring AI 1.0.9 and 1.1.8 were also released on June 12 as security patches.

**Security: CVE-2026-47835 (CVSS 8.6, June 2026):** Arbitrary query execution via special characters in Elasticsearch, OpenSearch, and GemFire vector store metadata filtering. Affects Spring AI versions before 1.0.9/1.1.8/2.0.0. Patch: upgrade to 1.0.9, 1.1.8, or 2.0.0 GA. If you use vector store metadata filters in production, patch immediately.

Spring has a rich history of successful standardization: Spring Data unified data access, Spring Integration standardized integration patterns, Spring Cloud simplified cloud development. Spring AI applies this proven approach to the world of AI and LLMs.

The key value of Spring AI lies in its deep integration with the Spring ecosystem. It is not a standalone library but an organic part of the Spring platform. Auto-configuration, dependency injection, configuration profiles, actuator metrics: all of these work with AI components just as naturally as with any other Spring beans.

## Project Philosophy

### Provider Portability

Spring AI declares portability as a first-class value. An application written for OpenAI should be able to switch to Anthropic, Azure, or a local model by changing configuration alone, without modifying code.

This is achieved through careful interface design. ChatModel, EmbeddingModel, ImageModel, and other interfaces define contracts implemented for each provider. Business logic operates against interfaces, and specific implementations are injected by Spring.

Portability is especially valuable in the rapidly evolving AI market. New models emerge regularly, and the ability to switch quickly provides a competitive advantage.

### Spring-Idiomatic Design

Spring AI follows Spring idioms and patterns. If you know Spring, you know Spring AI. Configuration through properties, conditional beans, profiles, testing: everything works in a familiar way.

This lowers the entry barrier for Spring developers. Instead of learning a new framework, they apply familiar concepts to a new domain.

### Production Readiness

Spring AI is developed with production use in mind. This means attention to reliability, observability, and performance.

Integration with Micrometer provides metrics. Structured logging simplifies debugging. Retry policies ensure resilience against transient provider failures.

## Spring AI Architecture

### Model API

The core of Spring AI consists of model abstractions. ChatModel represents conversational AI. EmbeddingModel handles embedding generation. ImageModel covers image generation.

Each abstraction defines a minimal contract common to all providers. Provider-specific capabilities are accessible through options and extensions.

Models are created as Spring beans and automatically configured from properties. You declare a dependency on ChatModel, and Spring injects the appropriate implementation.

### Prompt API

Prompts in Spring AI are typed objects, not just strings. This enables structured work with prompts.

PromptTemplate supports templating with placeholders. SystemPromptTemplate and UserPromptTemplate separate different prompt types. ChatOptions allows configuring generation parameters.

Typed prompts simplify testing and reuse. A prompt can be extracted into a separate file, loaded from configuration, and parameterized.

### Structured Output

Spring AI provides powerful mechanisms for obtaining structured responses from language models, solving one of the key problems of LLM applications, unpredictable response format.

**Evolution of Structured Output:**

Before the advent of specialized APIs, developers faced significant challenges. They had to add prompt instructions like "return JSON in the format...", then hope the model would correctly interpret the request, write parsing with error handling, and implement retry logic for invalid formats. The successful parsing rate was 85-90 percent, requiring complex exception handling.

Since August 2024, OpenAI provides an API for guaranteed structured responses, changing the paradigm. Instead of prompt instructions, the developer passes a JSON Schema in the response_format parameter, and the model is guaranteed to return valid JSON matching the schema. This is achieved through constrained decoding at the model level: tokens are generated with constraints that prevent structural violations.

Spring AI intelligently selects a strategy based on the provider's available capabilities. For OpenAI with Structured Outputs API support, the native mechanism is used with a 100 percent guarantee and minimal latency. For providers like Anthropic or Google, function calling is employed as an intermediate technique with approximately 99 percent reliability. For other models, the classic prompt-based approach with parsing and 90 percent success rate remains.

This adaptability makes code portable: the same business logic works across different providers, automatically using the best available structuring method.

### Advisors

Advisors are request and response interceptors that allow modifying the behavior of LLM calls. The concept is directly inspired by Spring AOP and Spring Security, applying a familiar pattern to a new domain.

**Advisors as AOP for Artificial Intelligence:**

The analogy with traditional AOP is very direct. In classic Spring AOP, you use annotations like @Around to intercept business logic method execution, applying cross-cutting concerns, logging, security, transactions. In Spring AI, Advisors intercept calls to language models in exactly the same way.

The join point in Spring AOP is method execution. In Spring AI, it is the LLM call. Around advice wraps execution. This gives control before and after. An Advisor does the same with the prompt (before) and the response (after). A pointcut defines which methods to intercept. In the AI context, this is the advisor chain configuration for a specific ChatClient.

**Cross-Cutting Concerns Reimagined for AI:**

Classic enterprise concerns translate elegantly to LLM applications. Logging is implemented via LoggingAdvisor instead of LoggingAspect. Security is provided by SafeGuardAdvisor, filtering prompt injections and unwanted content, replacing SecurityAspect. Prompt caching is handled by PromptCacheAdvisor. Retry logic for handling transient provider failures operates through RetryAdvisor.

Unique AI-specific concerns also emerge. QuestionAnswerAdvisor implements RAG, automatically enriching prompts with relevant context from a vector store. This has no counterpart in traditional AOP. It is a new category of functionality specific to LLMs.

**Composition and Execution Order:**

Advisors are composed into a chain when creating a ChatClient. Order matters: first QuestionAnswerAdvisor adds context from the knowledge base, then SafeGuardAdvisor checks the resulting prompt for safety, and finally LoggingAdvisor logs everything for auditing. Each advisor can modify the request, the response, or even halt processing if a problem is detected.

This architecture makes behavior extension declarative and modular, fully in line with Spring philosophy.

### Vector Store

Spring AI provides VectorStore, a vector storage abstraction. Implementations exist for Pinecone, Milvus, Qdrant, pgvector, Redis, Chroma, and others.

A unified interface allows switching stores without changing code. For development, InMemory can be used; for production: a distributed store.

DocumentReader and DocumentTransformer form the document processing pipeline before indexing.

### MCP Support (2025)

Spring AI 1.0 includes MCP (Model Context Protocol) client and server support:

**MCP Client:** Spring AI applications can connect to MCP servers to discover and use external tools and resources. MCP tools are automatically registered as Spring AI function callbacks. This makes them available to ChatClient without additional wiring.

**MCP Server:** Spring Boot applications can expose their own capabilities as MCP servers. Spring beans annotated with `@Tool` are automatically published as MCP tools. Resources and prompts are similarly exposed through standard Spring patterns.

**Transport:** Supports both stdio (for local integrations like Claude Desktop) and Streamable HTTP (for remote server deployments). Auto-configuration creates the appropriate transport based on application properties.

This means a Spring AI application can simultaneously act as an MCP client (consuming tools from external servers) and an MCP server (exposing its own tools to AI clients like Claude Desktop or Cursor).

## Configuration and Auto-Configuration

### Properties-Driven Configuration

Spring AI is fully configured through standard Spring mechanisms: application.properties or YAML files. You define the provider, model, and generation parameters declaratively using structured configuration keys.

A typical configuration includes the provider API key (usually via environment variables for security), the specific model selection (e.g., gpt-5.6 or claude-sonnet-5), and generation parameters such as temperature for controlling creativity, max-tokens for limiting length, or top-p for nucleus sampling.

This enables the powerful Spring profiles mechanism for different environments. Local development can use free Ollama with local models. A staging environment can use Azure OpenAI for corporate security. Production can use OpenAI or Anthropic directly for the best quality. Switching between environments requires only changing the active profile; the code remains unchanged.

### Conditional Beans

Spring AI auto-configuration creates beans conditionally. If the classpath contains an OpenAI dependency and an API key is configured, an OpenAI ChatModel is created. If Anthropic is present, an AnthropicChatModel is created.

Multiple providers can coexist simultaneously, distinguished by qualifiers. This is useful for fallback scenarios or specialized tasks.

### Customization

When deep customization is needed, you can define your own beans. Spring AI uses the standard @ConditionalOnMissingBean mechanism: your bean takes priority over the auto-configured one.

This allows fine-tuning clients, adding interceptors, and modifying behavior.

## Function Calling

### Defining Functions

Spring AI turns ordinary methods into tools. A method annotated with `@Tool` carries a description for the model, and `@ToolParam` describes each argument.

The tool description is taken from the annotation and passed to the model. The model can invoke a function by passing arguments as JSON. Spring AI deserializes the arguments and calls the method.

### Registration

Tools are registered explicitly. Pass a `@Tool`-annotated object to `defaultTools()` on the `ChatClient` builder to make it available on every request, or to `tools()` for a single call; per-call tools are added to the defaults. Defining the class as a Spring bean is not enough on its own, which keeps destructive tools out of requests that should not see them.

This still fits the Spring approach to development: tools are grouped into components, tested in isolation, and injected through DI.

### Execution Control

Spring AI provides control over function execution. You can restrict the set of available functions for a specific request. You can receive a function call without automatic execution for manual handling.

This is important for security, not all functions should be accessible to all users.

## Retrieval Augmented Generation

### RAG Components

Spring AI provides a complete set of components for RAG. DocumentReader loads documents in formats. DocumentSplitter breaks them into chunks. EmbeddingModel generates embeddings. VectorStore stores and searches.

QuestionAnswerAdvisor combines these components, automatically enriching queries with relevant context.

### Document Pipeline

Document loading and processing is organized as a pipeline. Documents pass through readers, transformers, and splitters, and are enriched with metadata.

The pipeline is configured declaratively. A typical setup includes the document format, chunk size, and splitting strategy.

### Search and Ranking

Spring AI supports search strategies. Similarity search is the basic vector search. Hybrid search combines it with keyword search. Metadata filtering filters by document attributes.

Search results are ranked by relevance. Thresholds and the number of returned documents can be configured.

## Observability

### Micrometer Integration

Spring AI integrates with Micrometer for metrics collection. Automatically collected metrics include: request count, latency, tokens, and errors.

Metrics are exported to any Micrometer-supported system: Prometheus, Datadog, CloudWatch, and others.

### Logging

Structured logs contain request context: identifier, model, and parameters. This simplifies correlation and debugging.

Logging levels are configured in the standard way. DEBUG shows prompts and responses. INFO covers main operations. WARN and ERROR indicate problems.

### Tracing

Integration with Spring Cloud Sleuth and Micrometer Tracing provides distributed tracing. AI calls become spans in the overall trace.

This is critical for understanding behavior in production, where AI is part of a larger workflow.

## Practical Architecture

A full-fledged Spring AI application combines several key components into a unified system. At the configuration level, a ChatClient is created with an advisor chain that includes QuestionAnswerAdvisor for RAG integration. This provides automatic prompt enrichment with context from the vector store.

Tool calling is implemented through methods annotated with `@Tool` on a regular Spring component, registered on the `ChatClient` with `defaultTools()` or per call with `tools()`. For example, a weather retrieval tool is a method with typed parameters that returns a record. The model invokes it by passing arguments as JSON, and Spring AI deserializes them, executes the call, and returns the result to the model.

The REST API is implemented using a standard Spring MVC controller. A basic chat endpoint accepts text, passes it to the ChatClient, and returns the response. A structured output endpoint uses the entity() method for automatic deserialization of the response into a typed class.

The configuration creates a ChatClient via a builder specifying a VectorStore for RAG. This pattern demonstrates how dependency injection, provider auto-configuration, standard testing through Spring Test, and observability through Micrometer work as a unified ecosystem.

## API Evolution

**Pre-1.0 to 1.0 changes:** The API stabilized for the 1.0 GA release. Key renames: `EmbeddingClient` -> `EmbeddingModel` (aligning with the `*Model` naming convention), `AiClient` -> `ChatClient` (early alpha rename). The `ChatClient` fluent API became the primary entry point, replacing direct `ChatModel.call()` for most use cases.

**ChatClient vs ChatModel:** `ChatModel` is the low-level interface (provider implementations). `ChatClient` is the high-level fluent API with advisor chains, structured output, and function calling integration. Production code should use `ChatClient`; `ChatModel` is for framework extensions and custom providers.

## Key Takeaways

Spring AI 2.0 GA (June 2026) is the current release: requires Spring Boot 4.0, adds progressive tool disclosure (ToolSearchToolCallingAdvisor), and enhanced observability. Check CVE-2026-47835 (CVSS 8.6) for vector store metadata filtering vulnerability, patch to 1.0.9/1.1.8/2.0.0.

Spring AI brings modern AI capabilities to the Spring ecosystem in a natural and idiomatic way. Provider portability, deep Spring integration, and production readiness make it a strong choice for enterprise applications.

Key advantages: unified model abstractions, typed prompts and structured output, the advisor pattern for extensibility, full-featured RAG, MCP client/server support, and deep observability.

For teams already working with Spring, Spring AI is a natural extension of the platform. Familiar patterns and tools are applied to a new domain with a minimal entry barrier.

## Practical Code Examples

### Example 1: ChatClient with Fluent API

```java
@RestController
@RequestMapping("/api/chat")
public class ChatController {

    private final ChatClient chatClient;

    public ChatController(ChatClient.Builder chatClientBuilder) {
        // Creating a ChatClient with default settings
        this.chatClient = chatClientBuilder
            .defaultSystem("You are a professional assistant. Respond concisely and to the point.")
            .build();
    }

    @PostMapping("/simple")
    public String simpleChat(@RequestBody String userMessage) {
        // Basic call with fluent API
        return chatClient.prompt()
            .user(userMessage)
            .call()
            .content();
    }

    @PostMapping("/parametrized")
    public String parametrizedChat(@RequestParam String topic,
                                   @RequestParam String style) {
        // Using parameterized prompts
        return chatClient.prompt()
            .user(u -> u.text("""
                Tell me about the topic: {topic}
                Presentation style: {style}
                """)
                .param("topic", topic)
                .param("style", style))
            .call()
            .content();
    }

    @PostMapping("/structured")
    public ProductReview getStructuredResponse(@RequestParam String reviewText) {
        // Obtaining a structured response with automatic deserialization
        return chatClient.prompt()
            .user("Analyze the review and extract structured information: " + reviewText)
            .call()
            .entity(ProductReview.class);
    }
}

// Record for structured output
record ProductReview(
    String sentiment,      // positive, negative, neutral
    int rating,            // rating from 1 to 5
    List<String> keyPoints // key points from the review
) {}
```

### Example 2: QuestionAnswerAdvisor Implementation for RAG

```java
@Configuration
public class RagConfiguration {

    @Bean
    public ChatClient ragChatClient(
            ChatClient.Builder builder,
            VectorStore vectorStore) {

        // Creating a QuestionAnswerAdvisor for automatic context enrichment
        QuestionAnswerAdvisor qaAdvisor = QuestionAnswerAdvisor.builder(vectorStore)
            .searchRequest(SearchRequest.builder()
                .topK(5)                    // Top 5 most relevant documents
                .similarityThreshold(0.7)   // Similarity threshold of 0.7
                .build())
            .build();

        return builder
            .defaultAdvisors(qaAdvisor)  // Adding the advisor to the chain
            .build();
    }
}

@Service
public class KnowledgeBaseService {

    private final ChatClient ragChatClient;
    private final VectorStore vectorStore;
    private final EmbeddingModel embeddingModel;

    public KnowledgeBaseService(ChatClient ragChatClient,
                               VectorStore vectorStore,
                               EmbeddingModel embeddingModel) {
        this.ragChatClient = ragChatClient;
        this.vectorStore = vectorStore;
        this.embeddingModel = embeddingModel;
    }

    // Indexing documents into the vector store
    public void indexDocuments(List<String> documents) {
        List<Document> docs = documents.stream()
            .map(content -> new Document(content))
            .toList();

        vectorStore.add(docs);
    }

    // Query with automatic RAG via QuestionAnswerAdvisor
    public String askQuestion(String question) {
        return ragChatClient.prompt()
            .user(question)
            .call()
            .content();
        // QuestionAnswerAdvisor automatically finds relevant documents
        // and adds them to the context before sending the request to the model
    }
}
```

### Example 3: Tool Calling - Invoking External Functions

```java
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.ai.tool.annotation.ToolParam;
import org.springframework.stereotype.Component;
import org.springframework.web.bind.annotation.*;

@Component
public class AssistantTools {

    @Tool(description = "Gets the current weather for the specified city")
    public WeatherResponse getCurrentWeather(
            @ToolParam(description = "City name") String city,
            @ToolParam(description = "Country code, for example DE") String country) {
        // Simulating a real Weather API call
        return new WeatherResponse(city, 22.5, "Sunny", 65);
    }

    @Tool(description = "Converts currency at the current exchange rate")
    public CurrencyResponse convertCurrency(
            @ToolParam(description = "Amount to convert") double amount,
            @ToolParam(description = "Source currency code, for example EUR") String from,
            @ToolParam(description = "Target currency code, for example USD") String to) {
        double rate = getExchangeRate(from, to);
        return new CurrencyResponse(amount, from, amount * rate, to, rate);
    }

    private double getExchangeRate(String from, String to) {
        // Stub for the example: EUR to USD
        return 1.08;
    }
}

// Record classes for typed tool results
record WeatherResponse(String city, double temperature, String condition, int humidity) {}
record CurrencyResponse(double originalAmount, String fromCurrency,
                       double convertedAmount, String toCurrency, double rate) {}

@RestController
@RequestMapping("/api/assistant")
public class AssistantController {

    private final ChatClient chatClient;

    public AssistantController(ChatClient.Builder builder, AssistantTools tools) {
        this.chatClient = builder
            .defaultSystem("You are a helpful assistant with access to weather and currency conversion tools.")
            // Tools are registered explicitly; a bean on the classpath is not enough
            .defaultTools(tools)
            .build();
    }

    @PostMapping("/query")
    public String processQuery(@RequestBody String userQuery) {
        // The model decides whether a tool call is needed,
        // and Spring AI executes it and returns the result to the model
        return chatClient.prompt()
            .user(userQuery)
            .call()
            .content();
    }
}
```
