```mermaid
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
	__start__([<p>__start__</p>]):::first
	selection(selection)
	answering(answering)
	check(check)
	__end__([<p>__end__</p>]):::last
	__start__ --> selection;
	answering --> check;
	selection --> answering;
	check -. &nbsp;True&nbsp; .-> __end__;
	check -. &nbsp;False&nbsp; .-> selection;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc
```
