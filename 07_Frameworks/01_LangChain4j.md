# LangChain4j: LLM Framework for Java

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Frameworks
**Previous:** [[00_Frameworks_Overview|AI Frameworks Overview]]
**Next:** [[02_Spring_AI|Spring AI]]

---

## Introduction to LangChain4j

LangChain4j is a full-featured framework for developing applications based on language models within the Java ecosystem. The project emerged as a response to the need of Java developers for tools comparable to the Python ecosystem, where LangChain has long been the de facto standard.

The philosophy of LangChain4j is built on several key principles. First, the framework strives to provide a unified interface for working with various LLM providers — from OpenAI to local models via Ollama. A developer can switch models by changing a single configuration line rather than rewriting the entire codebase.

Second, LangChain4j relies on a declarative approach through annotations. Instead of imperative code describing each step of interaction with the model, the developer declares interfaces with annotations, and the framework generates the implementation.

Third, the project actively integrates with the Spring ecosystem, providing auto-configuration and starters for Spring Boot. This makes incorporating LLMs into existing Spring applications as straightforward as possible.

## Architecture and Core Components

### Models and Providers

The central element of LangChain4j is the language model abstraction. The ChatLanguageModel interface defines the contract for interacting with any model that supports the chat-completion API.

The framework provides implementations for numerous providers: OpenAI, Azure OpenAI, Anthropic, Google Gemini, Amazon Bedrock, Mistral, Hugging Face, Ollama, and others. Each implementation adapts the provider's specifics to a unified interface.

Beyond chat models, LangChain4j supports embedding models through the EmbeddingModel interface, moderation models, image generation models, and other specialized types.

### AI Services

AI Services are the flagship feature of LangChain4j, enabling declarative interaction with LLMs through Java interfaces.

A developer creates an interface with methods representing the desired operations. Annotations describe prompts, system messages, and expected response formats. The framework generates an implementation that constructs requests to the model and parses responses.

This approach drastically reduces boilerplate code. Instead of dozens of lines for constructing a prompt, calling the API, and parsing the response — a single interface method with annotations.

### Memory and State

To support multi-turn conversations, LangChain4j provides the ChatMemory abstraction. Memory stores the message history and automatically includes it in each request to the model.

**Memory Types:**

**Conversation Memory (short-term)** answers the question "what were we just discussing?" and stores the message sequence of the current session. MessageWindowChatMemory retains the last N messages, TokenWindowChatMemory limits memory by token count, and SummarizingChatMemory periodically compresses history through summarization to fit within the model's context window.

**Semantic Memory (long-term)** answers the question "what has the user ever mentioned about topic X?" and stores user facts and preferences. This information is stored in a vector store, retrieval is performed by semantic relevance, and the memory is not tied to a specific session.

**When to use which**: for a standard chatbot within a single session, use MessageWindow for current conversation context. For a personal assistant, combine Semantic memory for long-term preferences with Conversation memory for short-term context. In technical support, use Conversation memory for ticket context. For application personalization, use Semantic memory to store user preferences.

Several memory implementations cover typical scenarios. Memory is associated with a session or user identifier, enabling independent conversations.

### RAG Components

LangChain4j includes a complete set of components for building RAG systems. EmbeddingStore abstracts vector stores: Pinecone, Milvus, Qdrant, Chroma, pgvector, and in-memory implementations are supported.

DocumentSplitter provides document splitting strategies. ContentRetriever organizes relevant content retrieval. RetrievalAugmentor combines components into a ready-made RAG pipeline.

RAG integration with AI Services is achieved through an annotation — simply specify a ContentRetriever, and the framework automatically enriches prompts with the retrieved context.

### Tools and Function Calling

LangChain4j supports tools — functions that the model can invoke to perform actions or retrieve information.

Tools are defined as methods of a Java class annotated with @Tool. The method and parameter descriptions are extracted from annotations and passed to the model. When a tool is invoked, the framework automatically routes the call to the corresponding method.

This mechanism enables models to interact with the external world: querying data from APIs, performing calculations, managing systems.

## Declarative AI Services

### Philosophy of the Declarative Approach

Traditional imperative code for working with LLMs quickly becomes cumbersome. Each request requires constructing a message list, calling the API, processing the response, and handling errors. When adding memory, tools, and RAG, the code grows exponentially.

AI Services offer an alternative: the developer describes what they want, not how to do it. The interface becomes the contract, and the framework handles the implementation.

**Why the declarative approach is better for AI:**

In the imperative style, you create a message list, add a system message with expert instructions, add user input, check for memory and add history if available, call the model for generation, save the response to memory, and parse the result. This leads to boilerplate code, scattered logic, code that is hard to test, and violation of the single responsibility principle.

In the declarative style, you define an interface with a SystemMessage annotation containing instructions and a method with a UserMessage annotation for the query parameter. This provides a clean abstraction that clearly expresses intent, is easily testable through interface substitution, and adheres to separation of concerns.

**Relationship with SOLID principles:**

Single Responsibility — the interface is responsible only for the contract, not the implementation. Open/Closed — extension through annotations without modifying code. Liskov Substitution — the implementation can be replaced. Interface Segregation — small, specialized interfaces. Dependency Inversion — dependency on abstraction, not implementation.

This approach not only reduces code but also makes the developer's intentions explicit. By reading the interface, one can understand all the capabilities of the AI service without studying implementation details.

### System and User Prompts

The @SystemMessage annotation defines the system prompt that sets the role and behavior of the model. This message is included at the beginning of each request and establishes the context for all subsequent interactions.

The user prompt is defined by the @UserMessage annotation on the method. It can contain a template with placeholders that are populated from method parameters.

The combination of system and user prompts allows precise control over model behavior for each operation.

### Structured Responses

AI Services support typed return values. If a method returns a specific class, the framework automatically instructs the model to return JSON and parses it into an object.

This elegantly solves the problem of extracting structured data from model responses. Instead of manually parsing text, the developer receives a ready-made object.

Complex types are supported: nested objects, collections, enums. The framework generates a JSON Schema from the return type and passes it to the model.

### Moderators and Guardrails

AI Services can include moderators — components that verify requests and responses against policies. A moderator can block undesirable content, filter sensitive information, and ensure compliance with corporate standards.

Moderators are integrated through annotations and executed automatically on each invocation.

## Spring Integration

### Spring Boot Starter

LangChain4j provides a Spring Boot Starter that automates configuration. Simply add the dependency and specify the API key in application.properties — the framework creates and configures all necessary beans.

The starter supports all major model providers, vector stores, and embedding models.

### Auto-Configuration

The Spring Boot auto-configuration mechanism allows LangChain4j to adapt to the environment. When certain dependencies are present, the corresponding beans are automatically created.

If the PostgreSQL driver is on the classpath and a connection is configured — a pgvector EmbeddingStore is created. If an OpenAI API key is configured — an OpenAI ChatLanguageModel is created.

### Dependency Injection

AI Services become Spring beans through annotation or programmatic registration. They automatically receive dependencies: model, memory, RAG components, and tools.

This provides natural integration into Spring applications. An AI service is injected like any other bean and used in controllers, services, and scheduled tasks.

## Working with Tools

### Defining Tools

Tools in LangChain4j are methods that the model can invoke. Each tool has a name, description, and parameters.

The @Tool annotation on a method makes it available to the model. The @P annotation on parameters describes their purpose. This information is passed to the model for invocation decisions.

It is important that descriptions are clear and complete. The model decides whether to invoke a tool based on the description, not the code.

### Dynamic Tools

The set of tools can be formed dynamically depending on the context. A user with certain permissions gets access to some tools, while a user without permissions gets access to others.

ToolProvider allows dynamically defining available tools for each request.

### Handling Results

The result of a tool invocation is automatically serialized and passed back to the model. The model uses this information to formulate the final response or decide on the next action.

Both synchronous and asynchronous tools are supported. Long-running operations can execute asynchronously without blocking the thread.

## Practical Architecture

The basic structure of a LangChain4j application includes three main elements: a declarative AI Service interface, a class with tools, and initialization code.

**The declarative AI Service interface** defines the contract for interaction with the model. Methods are annotated with @SystemMessage for context, @UserMessage for queries, and @MemoryId for binding to a user session. Return types can be strings for simple responses or specialized record classes for structured data that the framework automatically parses from the model's JSON response.

**The tools class** contains methods annotated with @Tool that are available to the model for invocation. Each method has a description of its purpose and parameters annotated with @P explaining their meaning. For example, a time retrieval tool returns the current date via LocalDateTime, and a search tool accepts a string query and returns results from the database.

**Initialization** creates the model via a builder specifying the API key and model name, configures memory with a message count limit, and assembles the AI Service through AiServices.builder, passing the model, memory, and tools class instance. After this, the service is ready for use — method calls automatically construct prompts, invoke the model, use tools when needed, and parse responses.

This pattern demonstrates the core capabilities: declarativeness, memory management, function calling, and structured output.

## Key Takeaways

LangChain4j is a mature and feature-rich framework that brings modern LLM development capabilities to the Java ecosystem. Unified interfaces abstract providers, declarative AI Services reduce boilerplate code, and Spring integration simplifies adoption.

Key advantages of the framework: broad provider support, declarative approach through annotations, complete set of RAG components, powerful tool system, and deep integration with Spring Boot.

For Java teams building AI applications, LangChain4j is the natural choice, combining the capabilities of modern LLMs with familiar Java patterns and tools.

## Navigation

**Module:** [[../Table_of_Contents|Course Contents]] → Frameworks
**Previous:** [[00_Frameworks_Overview|AI Frameworks Overview]]
**Next:** [[02_Spring_AI|Spring AI]]

---

## Practical Code Examples

### 1. Defining an AI Service with Annotations

```java
import dev.langchain4j.service.SystemMessage;
import dev.langchain4j.service.UserMessage;
import dev.langchain4j.service.MemoryId;

/**
 * AI Service for technical support
 * Demonstrates the use of system and user prompts
 */
public interface SupportAssistant {

    @SystemMessage("""
        You are a technical support assistant for TechCorp.
        Your task is to help users resolve technical issues.
        Respond politely, professionally, and to the point.
        If you do not know the answer, be honest and suggest contacting a specialist.
        """)
    String chat(@MemoryId String userId, @UserMessage String message);
}

/**
 * AI Service for extracting structured data
 * Demonstrates returning typed objects
 */
public interface DataExtractor {

    @SystemMessage("Extract information from the text and return it in a structured format.")
    @UserMessage("Extract person information from the text: {{text}}")
    PersonInfo extractPerson(String text);
}

/**
 * Record for structured data
 */
record PersonInfo(
    String name,
    Integer age,
    String occupation,
    String email
) {}
```

### 2. Defining Tools with @Tool

```java
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.P;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Class with tools for the agent
 * Demonstrates various types of tools
 */
public class AgentTools {

    /**
     * Tool for getting the current time
     */
    @Tool("Get the current date and time")
    public String getCurrentDateTime() {
        LocalDateTime now = LocalDateTime.now();
        DateTimeFormatter formatter = DateTimeFormatter.ofPattern("dd.MM.yyyy HH:mm:ss");
        return "Current time: " + now.format(formatter);
    }

    /**
     * Tool for searching product information
     */
    @Tool("Find product information by its name or SKU")
    public String searchProduct(
            @P("Product name or SKU") String query) {

        // Simulating a database search
        return switch (query.toLowerCase()) {
            case "laptop" ->
                "Laptop Pro X1: 1299 EUR, 15 units in stock. " +
                "Specs: 16GB RAM, 512GB SSD, Intel i7";
            case "mouse" ->
                "Wireless Mouse Elite: 49 EUR, 50 units in stock. " +
                "Wireless, ergonomic, 6 buttons";
            default ->
                "Product not found. Try a different query.";
        };
    }

    /**
     * Tool for calculating a discount
     */
    @Tool("Calculate the final price with a discount applied")
    public String calculateDiscount(
            @P("Original price in euros") double price,
            @P("Discount percentage (0-100)") int discountPercent) {

        if (discountPercent < 0 || discountPercent > 100) {
            return "Error: discount percentage must be between 0 and 100";
        }

        double discount = price * discountPercent / 100.0;
        double finalPrice = price - discount;

        return String.format(
            "Original price: %.2f EUR\n" +
            "Discount %d%%: %.2f EUR\n" +
            "Final price: %.2f EUR",
            price, discountPercent, discount, finalPrice
        );
    }
}
```

### 3. Full Integration with ChatLanguageModel

```java
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.service.AiServices;

/**
 * Class demonstrating the full setup and usage of LangChain4j
 */
public class LangChain4jDemo {

    public static void main(String[] args) {
        // 1. Creating a model with settings
        ChatLanguageModel model = OpenAiChatModel.builder()
                .apiKey(System.getenv("OPENAI_API_KEY"))
                .modelName("gpt-4o")
                .temperature(0.7)
                .maxTokens(1000)
                .logRequests(true)
                .logResponses(true)
                .build();

        // 2. Creating tools
        AgentTools tools = new AgentTools();

        // 3. Configuring memory (last 10 messages)
        MessageWindowChatMemory memory = MessageWindowChatMemory.withMaxMessages(10);

        // 4. Building the AI Service
        SupportAssistant assistant = AiServices.builder(SupportAssistant.class)
                .chatLanguageModel(model)
                .chatMemory(memory)
                .tools(tools)
                .build();

        // 5. Using the assistant
        String userId = "user123";

        // First request - the agent will use the getCurrentDateTime tool
        String response1 = assistant.chat(userId,
            "Hello! What time is it now?");
        System.out.println("Assistant: " + response1);

        // Second request - the agent will use the searchProduct tool
        String response2 = assistant.chat(userId,
            "Do you have any laptops in stock?");
        System.out.println("Assistant: " + response2);

        // Third request - the agent will use the calculateDiscount tool
        String response3 = assistant.chat(userId,
            "How much would the laptop cost with a 15% discount?");
        System.out.println("Assistant: " + response3);

        // 6. Example of extracting structured data
        demonstrateStructuredOutput(model);
    }

    /**
     * Demonstration of extracting structured data
     */
    private static void demonstrateStructuredOutput(ChatLanguageModel model) {
        DataExtractor extractor = AiServices.builder(DataExtractor.class)
                .chatLanguageModel(model)
                .build();

        String text = """
            Ivan Petrov, 35 years old, works as a software developer at TechCorp.
            His email: ivan.petrov@example.com
            """;

        PersonInfo person = extractor.extractPerson(text);

        System.out.println("\nExtracted data:");
        System.out.println("Name: " + person.name());
        System.out.println("Age: " + person.age());
        System.out.println("Occupation: " + person.occupation());
        System.out.println("Email: " + person.email());
    }
}
```

### Key Implementation Points

**Declarative approach**: The AI Service is defined as an interface with annotations. The framework automatically generates an implementation that handles prompts, model calls, and response parsing.

**Tools**: Methods annotated with `@Tool` automatically become available to the model for invocation. The model independently decides when and which tool to use based on descriptions.

**Memory**: `MessageWindowChatMemory` retains the last N messages to maintain conversation context. Memory is bound to `@MemoryId` for separate user sessions.

**Structured output**: By returning typed objects (record, POJO), you automatically get JSON parsing. The framework generates the JSON Schema and instructs the model on its own.

**Modularity**: Components (model, memory, tools) are created separately and assembled via the builder pattern, ensuring flexibility and testability.
