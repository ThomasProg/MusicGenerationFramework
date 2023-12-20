

3 different ways to do callbacks:

- C++ class inheritance / virtual functions
    - But not compatible with python etc
    - No problem with async

- C callbacks
    - Overhead?

- Polling
    - Requires structs, so have to convert anyways
    - What about async? mutex? have to call dll function anyways



Async:
- The melody should be generated async
- The player should be played in the game thread 
- Mutex: when the player accesses generated music + when the generation writes / switch buffer?

