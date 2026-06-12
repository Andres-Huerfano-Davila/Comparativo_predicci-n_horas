# Comparativo y predicción de horas nómina - V3

Herramienta Streamlit con dos módulos:

1. Comparativo histórico: pagado real vs provisión vs proyección vs headcount.
2. Predicción mes en curso: interfaces + histórico + headcount + MD actual + cuentas contables.

## Ajuste V3

El MD actual puede cargarse como TXT SAP completo. La app consolida salario vigente aplicando:

- Si una persona tiene `Hasta = 31.12.9999`, usa esa vigencia.
- Si no tiene vigencia abierta, usa la fecha máxima de `Hasta` por persona.
- Dentro de la vigencia usada, toma el último registro por `SAP + concepto` ordenando por `Modif.el` descendente y luego `Importe` descendente.
- Suma salario base + bonos para calcular `Salario total`.
- Calcula jornada vigente: si existe `H sem.`, usa `H sem. × 5`; si no, usa el parámetro de jornada por defecto.
- Calcula `valor_hora = salario_total / jornada`.
- Calcula costo estimado: `cantidad_estimada × valor_hora × factor_concepto`.
