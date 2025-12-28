# System Behavior & Threat Logic

This document describes the current threat detection logic triggered by the Inference Pipeline.

## Threat Logic Matrix

The system classifies threats based on a combination of **Action** (recognized by LSTM), **Object Context** (detected by YOLO), and **Duration** (time presence).

| Threat Level | Action (LSTM) | Object Context | Duration | Output Label | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **CRITICAL** | `shooting` \| `violence` | **Gun** | *Any* | `CRITICAL: SHOOTER` | Active shooter detected. |
| **CRITICAL** | `shooting` \| `violence` | **Knife** | *Any* | `CRITICAL: KNIFE ATTACK` | Active knife attack. |
| **CRITICAL** | `shooting` \| `violence` | **ATM** | *Any* | `CRITICAL: ATM ROBBERY` | Violence occurring near an ATM. |
| **HIGH** | *Any* | **Gun** \| **Knife** | *Any* | `HIGH: WEAPON DETECTED` | Weapon visible. |
| **WARNING** | `normal` | **ATM** | **> 60s** | `WARN: LOITERING AT ATM` | Person present at ATM for prolonged time. |
| **WARNING** | *No Person* | **Bag** (Backpack/Handbag) | **> 60s** | `WARN: UNATTENDED BAGGAGE` | Bag visible with no person for prolonged time. |
| **WARNING** | `shooting` | *No Weapon* | *Any* | `WARN: SUSPICIOUS STANCE` | Shooting motion only. |
| **WARNING** | `violence` | *No Weapon* | *Any* | `WARN: FIGHTING` | Fighting motion only. |
| **SAFE** | `normal` | *Any* | **< 60s** | `SAFE` | Safe condition. |

## Thresholds

- **ATM Loitering**: 60.0 seconds.
- **Unattended Baggage**: 60.0 seconds.
