# PrometheusLLM Architecture Diagram

The following diagrams illustrate the architecture of the PrometheusLLM transformer model.

## Overall Architecture

```mermaid
graph TB
    subgraph "PrometheusLLM Model"
        Input["Input Tokens"] --> InputEmb["Input Embedding"]
        InputEmb --> PE1["Positional Encoding"]
        PE1 --> Encoder["Encoder"]
        Encoder --> CrossAttention
        
        TgtInput["Target Tokens"] --> TgtEmb["Target Embedding"]
        TgtEmb --> PE2["Positional Encoding"]
        PE2 --> Decoder["Decoder"]
        CrossAttention --> Decoder
        
        Decoder --> OutProj["Output Projection"]
        OutProj --> Output["Output Probabilities"]
    end
    
    Output --> Generation["Text Generation"]
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef highlight fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    class Encoder,Decoder highlight
```

## Encoder Architecture

```mermaid
graph TB
    subgraph "Encoder"
        EIn["Encoder Input"] --> EL1["Encoder Layer 1"]
        EL1 --> EL2["Encoder Layer 2"]
        EL2 --> ELN["..."]
        ELN --> ELast["Encoder Layer N"]
        ELast --> EOut["Encoder Output"]
    end
    
    subgraph "Encoder Layer"
        ELIn["Layer Input"] --> SelfAttn["Self-Attention"]
        SelfAttn --> AddNorm1["Add & Norm"]
        AddNorm1 --> FFN["Feed Forward Network"]
        FFN --> AddNorm2["Add & Norm"]
        AddNorm2 --> ELOut["Layer Output"]
    end
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef highlight fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    class SelfAttn highlight
```

## Decoder Architecture

```mermaid
graph TB
    subgraph "Decoder"
        DIn["Decoder Input"] --> DL1["Decoder Layer 1"]
        DL1 --> DL2["Decoder Layer 2"]
        DL2 --> DLN["..."]
        DLN --> DLast["Decoder Layer N"]
        DLast --> DOut["Decoder Output"]
    end
    
    subgraph "Decoder Layer"
        DLIn["Layer Input"] --> MaskedSelfAttn["Masked Self-Attention"]
        MaskedSelfAttn --> AddNorm1["Add & Norm"]
        AddNorm1 --> CrossAttn["Cross-Attention"]
        CrossAttn --> AddNorm2["Add & Norm"]
        AddNorm2 --> FFN["Feed Forward Network"]
        FFN --> AddNorm3["Add & Norm"]
        AddNorm3 --> DLOut["Layer Output"]
    end
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef highlight fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    class MaskedSelfAttn,CrossAttn highlight
```

## Multi-Head Attention Architecture

```mermaid
graph TB
    subgraph "Multi-Head Attention"
        Q["Query"] --> QProj["Q Linear Projection"]
        K["Key"] --> KProj["K Linear Projection"]
        V["Value"] --> VProj["V Linear Projection"]
        
        QProj --> Head1["Attention Head 1"]
        KProj --> Head1
        VProj --> Head1
        
        QProj --> Head2["Attention Head 2"]
        KProj --> Head2
        VProj --> Head2
        
        QProj --> HeadN["Attention Head N"]
        KProj --> HeadN
        VProj --> HeadN
        
        Head1 --> Concat["Concatenate"]
        Head2 --> Concat
        HeadN --> Concat
        
        Concat --> OutProj["Output Linear Projection"]
        OutProj --> Out["Attention Output"]
    end
    
    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef highlight fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    class Head1,Head2,HeadN highlight
```

## Connection to EdenCore

```mermaid
graph LR
    subgraph "EdenCore Architecture"
        CoreState["State Vector"] --> CoreProj["Projection Layer"]
        CoreProj --> CoreTrans["Transformer Block"]
        CoreTrans --> CoreBack["Projection Back"]
        CoreBack --> CoreState
    end
    
    subgraph "PrometheusLLM Architecture"
        Input["Input Tokens"] --> Embedding["Token Embedding"]
        Embedding --> Encoder["Encoder Layers"]
        Encoder --> Decoder["Decoder Layers"]
        Decoder --> OutProj["Output Projection"]
        OutProj --> Output["Output Tokens"]
    end
    
    CoreState -.Inspires.-> Input
    CoreProj -.Inspires.-> Embedding
    CoreTrans -.Extends To.-> Encoder
    CoreBack -.Inspires.-> OutProj
    
    classDef eden fill:#f8d7da,stroke:#721c24,stroke-width:2px;
    classDef prometheus fill:#d4edda,stroke:#155724,stroke-width:2px;
    class CoreState,CoreProj,CoreTrans,CoreBack eden;
    class Input,Embedding,Encoder,Decoder,OutProj,Output prometheus;
```
