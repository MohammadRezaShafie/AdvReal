# Training Process Flowchart

Complete step-by-step visualization of the AdvReal adversarial patch training process.

---

## 🔄 High-Level Training Loop

```mermaid
flowchart TB
    START([Start Training]) --> INIT[Initialize Components]
    INIT --> EPOCH_LOOP{For each epoch<br/>0 to nepoch}
    
    EPOCH_LOOP -->|epoch < nepoch| BATCH_LOOP{For each batch}
    BATCH_LOOP -->|has batches| PROCESS[Process Batch]
    PROCESS --> BATCH_LOOP
    
    BATCH_LOOP -->|done| LOG[Log Epoch Metrics]
    LOG --> SAVE{epoch % 10 == 0?}
    SAVE -->|yes| SAVE_PATCH[Save Patch Image]
    SAVE -->|no| NEXT
    SAVE_PATCH --> NEXT[Next Epoch]
    NEXT --> EPOCH_LOOP
    
    EPOCH_LOOP -->|done| END([Training Complete])
```

---

## 📦 Initialization Phase

```mermaid
flowchart TB
    subgraph INIT ["🔧 Initialization"]
        direction TB
        A1[Parse Arguments] --> A2[Load Config YAML]
        A2 --> A3[Create PatchTrainer]
        
        A3 --> M1[Load Detector Model<br/>YOLOv2/v3/v5/v11/RCNN/DETR]
        A3 --> M2[Load 3D Meshes<br/>man.obj, tshirt.obj, trouser.obj]
        A3 --> M3[Initialize Renderer<br/>ImageRenderer]
        
        M1 --> L1[Create Loss Extractor<br/>MaxProbExtractor]
        M2 --> L2[Setup PyTorch3D<br/>Cameras, Lights, Rasterizer]
        M3 --> L3[Load Background Images<br/>background_loader]
        
        L1 --> P1[Initialize Universal Patch<br/>gray/random/white]
        L2 --> P1
        L3 --> P1

        P1 --> O1[Create Adam Optimizer<br/>lr=0.03, amsgrad=True]
        O1 --> DONE[Ready for Training]
    end
```

---

## 🔁 Single Batch Processing (Core Loop)

```mermaid
flowchart TB
    subgraph BATCH ["📦 Process Single Batch"]
        direction TB
        
        B1[/"Load bg_batch from<br/>background_loader"/] --> B2[/"Load person_img_batch from<br/>person_detection_loader"/]
        
        B2 --> ZERO[optimizer.zero_grad]
        
        ZERO --> D2D
        
        subgraph D2D ["🎯 2D Detection Attack"]
            direction TB
            D1[Detect persons in<br/>person_img_batch] --> D2[Get bounding boxes<br/>all_preds]
            D2 --> D3{Any targets<br/>detected?}
            D3 -->|no| SKIP[Skip batch]
            D3 -->|yes| D4[Apply patch to<br/>person images]
            D4 --> D5[Compute 2D losses<br/>patch_det_loss, patch_tv_loss]
        end
        
        D5 --> R3D
        
        subgraph R3D ["🎨 3D Rendering Pipeline"]
            direction TB
            R1[Clone & clamp patch<br/>patch_c = patch.clamp 0,1] --> R2[Set patch as texture<br/>renderer.set_adv_patch_texture]
            R2 --> R3[For each background image]
            R3 --> R4[Render 3D person<br/>with patch texture]
            R4 --> R5[Composite onto background<br/>generate_composite_image_tensor]
            R5 --> R6[Stack all composites<br/>p_img_batch]
        end
        
        R6 --> DET3D
        
        subgraph DET3D ["🔍 3D Detection"]
            direction TB
            T1[Normalize images<br/>to 0-1 range] --> T2[Run detector<br/>model forward pass]
            T2 --> T3[Extract confidence scores<br/>prob_extractor]
            T3 --> T4[Compute 3D detection loss<br/>det_loss]
        end
        
        T4 --> LOSS[Compute Total Loss<br/>L = det_loss + patch_det_loss + patch_tv_loss]
        
        LOSS --> BP
        
        subgraph BP ["⬅️ Backpropagation"]
            direction TB
            B1B[loss.backward] --> B2B[optimizer.step]
            B2B --> B3B[patch.clamp 0,1]
        end
        
        BP --> LOG_B[Log to TensorBoard<br/>every 10 batches]
    end
```

---

## 🖼️ Image Transformation Pipeline

```mermaid
flowchart LR
    subgraph INPUT ["📥 Inputs"]
        I1[Person Images<br/>416×416]
        I2[Background Images<br/>variable size]
        I3[Adversarial Patch<br/>300×300]
    end
    
    subgraph PATCH_APPLY ["🩹 2D Patch Application"]
        P1[Detect person bbox] --> P2[Scale patch to bbox]
        P2 --> P3[Apply transforms<br/>jitter, rotate, median_pool]
        P3 --> P4[Overlay patch on person]
    end
    
    subgraph RENDER ["🎨 3D Rendering"]
        R1[Load 3D mesh] --> R2[Apply patch as UV texture]
        R2 --> R3[Setup camera angles]
        R3 --> R4[Apply TPS deformation<br/>NRSM cloth simulation]
        R4 --> R5[Render with lighting]
        R5 --> R6[Generate alpha mask]
    end
    
    subgraph COMPOSITE ["🖼️ Compositing"]
        C1[Random crop position<br/>in background] --> C2[Apply relighting<br/>match lighting conditions]
        C2 --> C3[Blend rendered person<br/>with background]
        C3 --> C4[Final composite image]
    end
    
    I1 --> PATCH_APPLY
    I3 --> PATCH_APPLY
    I3 --> RENDER
    I2 --> COMPOSITE
    RENDER --> COMPOSITE
    
    PATCH_APPLY --> DET1[2D Detection Loss]
    COMPOSITE --> DET2[3D Detection Loss]
```

---

## 📉 Loss Computation Flow

```mermaid
flowchart TB
    subgraph LOSSES ["📉 Loss Computation"]
        direction TB
        
        subgraph L2D ["2D Losses"]
            L2D1[Run detector on<br/>patched 2D images] --> L2D2[Get confidence scores]
            L2D2 --> L2D3[patch_det_loss = mean conf]
            L2D3 --> L2D4[patch_tv_loss = TV of patch]
        end
        
        subgraph L3D ["3D Losses"]
            L3D1[Run detector on<br/>composite 3D images] --> L3D2[Match boxes with GT<br/>IoU threshold]
            L3D2 --> L3D3{loss_type?}
            L3D3 -->|max_iou| L3D4[score of max IoU box]
            L3D3 -->|max_conf| L3D5[maximum confidence]
            L3D3 -->|softplus_*| L3D6[smooth max with softplus]
            L3D4 --> L3D7[det_loss = mean over batch]
            L3D5 --> L3D7
            L3D6 --> L3D7
        end
        
        L2D4 --> TOTAL
        L3D7 --> TOTAL
        
        TOTAL[Total Loss = det_loss + patch_det_loss + patch_tv_loss]
    end
```

---

## 🔄 Optimization Loop Detail

```mermaid
flowchart TB
    subgraph OPT ["🔄 Patch Optimization"]
        direction TB
        
        PATCH[Adversarial Patch δ]
        
        PATCH --> FWD[Forward Pass]
        FWD --> LOSS[Compute Loss L]
        LOSS --> BWD[loss.backward<br/>Compute ∇L]
        
        BWD --> METHOD{Attack Method?}
        
        METHOD -->|optim| ADAM[Adam Update<br/>δ -= α × Adam momentum,variance]
        METHOD -->|pgd| PGD[PGD Update<br/>δ += α × sign ∇L]
        METHOD -->|bim| BIM[BIM Update<br/>δ += clip α × sign ∇L, ±ε]
        METHOD -->|mim| MIM[MIM Update<br/>g = μg + ∇L/norm<br/>δ += α × sign g]
        
        ADAM --> CLAMP[Clamp δ to 0,1]
        PGD --> CLAMP
        BIM --> CLAMP
        MIM --> CLAMP
        
        CLAMP --> PATCH
    end
```

---

## 📊 Complete Training Timeline

```mermaid
sequenceDiagram
    participant U as User
    participant T as PatchTrainer
    participant D as Detector
    participant R as Renderer
    participant O as Optimizer
    
    U->>T: Start train()
    T->>T: Initialize components
    
    loop For each epoch
        T->>T: Reset epoch metrics
        
        loop For each batch
            T->>T: Load bg_batch, person_img_batch
            T->>O: optimizer.zero_grad()
            
            Note over T,D: 2D Attack Phase
            T->>D: detect_bbox(person_img_batch)
            D-->>T: all_preds (bboxes)
            T->>T: Apply patch to images
            T->>D: attack() → get 2D loss
            D-->>T: patch_det_loss, patch_tv_loss
            
            Note over T,R: 3D Rendering Phase
            T->>R: set_adv_patch_texture(patch)
            T->>R: generate_composite_image_tensor()
            R-->>T: composite images + ground truth
            
            Note over T,D: 3D Detection Phase
            T->>D: model(composite_images)
            D-->>T: detector output
            T->>T: prob_extractor → det_loss
            
            Note over T,O: Backpropagation
            T->>T: total_loss = det_loss + 2D losses
            T->>O: loss.backward()
            T->>O: optimizer.step()
            T->>T: patch.clamp(0, 1)
            
            T->>T: Log to TensorBoard
        end
        
        T->>T: Save patch (every 10 epochs)
        T->>T: Print epoch summary
    end
    
    T->>U: Training Complete
```

---

## 📁 Data Flow Summary

```
┌──────────────────────────────────────────────────────────────────────┐
│                         TRAINING LOOP                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐          │
│  │ Person       │     │ Background   │     │ Adversarial  │          │
│  │ Images       │     │ Images       │     │ Patch        │          │
│  │ (INRIAPerson)│     │ (NuScenes)   │     │ (Learnable)  │          │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘          │
│         │                    │                    │                  │
│         ▼                    │                    │                  │
│  ┌──────────────┐            │                    │                  │
│  │ 2D Detection │◄───────────┼────────────────────┤                  │
│  │ + Patch Apply│            │                    │                  │
│  └──────┬───────┘            │                    │                  │
│         │                    │                    │                  │
│         ▼                    │                    ▼                  │
│  ┌──────────────┐            │           ┌──────────────┐            │
│  │ 2D Det Loss  │            │           │ 3D Rendering │            │
│  │ + TV Loss    │            │           │ (PyTorch3D)  │            │
│  └──────┬───────┘            │           └──────┬───────┘            │
│         │                    │                  │                    │
│         │                    ▼                  ▼                    │
│         │           ┌────────────────────────────────┐               │
│         │           │ Composite Images               │               │
│         │           │ (3D rendered + background)     │               │
│         │           └──────────────┬─────────────────┘               │
│         │                          │                                 │
│         │                          ▼                                 │
│         │                   ┌──────────────┐                         │
│         │                   │ 3D Detection │                         │
│         │                   │ Loss         │                         │
│         │                   └──────┬───────┘                         │
│         │                          │                                 │
│         ▼                          ▼                                 │
│  ┌────────────────────────────────────────────────────────┐          │
│  │         TOTAL LOSS = L_3D + L_2D_det + L_2D_TV         │          │
│  └────────────────────────────────────────────────────────┘          │
│                              │                                       │
│                              ▼                                       │
│                    ┌──────────────────┐                              │
│                    │ Backpropagation  │                              │
│                    │ + Optimizer Step │                              │
│                    └────────┬─────────┘                              │
│                              │                                       │
│                              ▼                                       │
│                    ┌──────────────────┐                              │
│                    │ Updated Patch    │──────────────┐               │
│                    └──────────────────┘              │               │
│                                                      │               │
│                              ▲                       │               │
│                              └───────────────────────┘               │
│                           (Next Iteration)                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Key Config Parameters

| Parameter | Location | Effect |
|-----------|----------|--------|
| `nepoch` | args | Number of training epochs |
| `batch_size` | args | Images per batch |
| `lr` | args | Adam learning rate |
| `tv_eta` | YAML | TV loss weight |
| `ATTACK_CLASS` | YAML | Target class ID |
| `loss_type` | args | Detection loss type |
| `train_iou` | args | IoU threshold for loss |
