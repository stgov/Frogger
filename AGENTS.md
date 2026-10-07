# Directrices para el Agente de Programación (Git-Céntrico y Determinista)

Este repositorio utiliza una arquitectura de **Shadow Git** y **Verificación Determinista** para garantizar trazabilidad sin interferir con el control de versiones principal del desarrollador.

---

## 1. Regla de Oro: Shadow Git como Fuente de Verdad
- **Aislamiento**: El repositorio Git principal (`.git`) pertenece al usuario. El agente **NO debe hacer commits ni modificar `.git`** a menos que se le pida de forma explícita.
- **Repositorio del Agente**: Todas las operaciones de control de versiones del agente se realizan en `.git-agent`.
- **Uso del Helper**: Ejecutar comandos usando `.\agent-git.ps1 <comando>` (o `git --git-dir=.git-agent --work-tree=. <comando>`).
- **Prohibido asumir el estado por contexto**:
  - Antes de empezar a trabajar o editar archivos, ejecuta `.\agent-git.ps1 status`.
  - Para entender el historial o contexto previo de cambios, ejecuta `.\agent-git.ps1 log -n 5 --oneline`.
  - Para verificar las modificaciones realizadas, ejecuta `.\agent-git.ps1 diff`.

---

## 2. Flujo de Trabajo y Commits Atómicos
1. **Inspeccionar**: Verificar el estado inicial con `.\agent-git.ps1 status`.
2. **Modificar**: Realizar los cambios puntuales en los archivos.
3. **Evaluar Diff**: Revisar el diff exacto con `.\agent-git.ps1 diff <archivo>`.
4. **Verificación Determinista**: Confirmar que no hay errores sintácticos (`ast` / `py_compile`) ni regresiones.
5. **Commit Atómico**: Hacer commit de los archivos específicos usando Conventional Commits:
   ```powershell
   .\agent-git.ps1 add <ruta-del-archivo>
   .\agent-git.ps1 commit -m "feat(modulo): descripcion concisa"
   ```
   *Tipos permitidos*: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`.

---

## 3. Hooks y Verificación Automática
- Antigravity ejecuta el hook `PostToolUse` configurado en `.agent/hooks.json` tras cada edición de archivos (`replace_file_content` o `write_to_file`), evaluando el árbol AST y el diff generado.
- Si recibes un mensaje `⚠️ [Deterministic Check FAILED]`, tu prioridad absoluta es corregir el fallo de sintaxis antes de avanzar a otra tarea.
- El repositorio `.git-agent` cuenta con un hook `pre-commit` que previene cualquier commit con errores de sintaxis en Python.
