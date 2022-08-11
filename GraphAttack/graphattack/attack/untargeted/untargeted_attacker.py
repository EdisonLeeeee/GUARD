from graphattack.attack.flip_attacker import FlipAttacker


class UntargetedAttacker(FlipAttacker):

    def reset(self) -> "UntargetedAttacker":
        """Reset the state of the Attacker

        Returns
        -------
        UntargetedAttacker
            the attacker itself
        """
        super().reset()
        self.num_budgets = None
        self.structure_attack = None
        self.feature_attack = None
        return self

    def attack(self, num_budgets, structure_attack, feature_attack) -> "UntargetedAttacker":
        """Base method that describes the adversarial untargeted attack

        Parameters
        ----------
        num_budgets : int (0<`num_budgets`<=:attr:max_perturbations) or float (0<`num_budgets`<=1)
            Case 1:
            `int` : the number of attack budgets, 
            i.e., how many edges can be perturbed.

            Case 2:
            `float`: the number of attack budgets is 
            the ratio of :attr:max_perturbations

            See `:attr:max_perturbations`

        structure_attack : bool
            whether to conduct structure attack, i.e., modify the graph structure (edges)
        feature_attack : bool
            whether to conduct feature attack, i.e., modify the node features

        """
        
        _is_setup = getattr(self, "_is_setup", True)
        
        if not _is_setup:
            raise RuntimeError(
                f'{self.__class__.__name__} requires a surrogate model to conduct attack. '
                'Use `attacker.setup_surrogate(surrogate_model)`.')        

        if not self.is_reseted:
            raise RuntimeError(
                'Before calling attack, you must reset your attacker. Use `attacker.reset()`.'
            )

        if not (structure_attack or feature_attack):
            raise RuntimeError(
                'Either `structure_attack` or `feature_attack` must be True.')

        if feature_attack and not self._allow_feature_attack:
            raise RuntimeError(
                f"{self.name} does NOT support attacking features."
                " If the model can conduct feature attack, please call `attacker.set_allow_feature_attack(True)`."
            )

        if structure_attack and not self._allow_structure_attack:
            raise RuntimeError(
                f"{self.name} does NOT support attacking structures."
                " If the model can conduct structure attack, please call `attacker.set_allow_structure_attack(True)`."
            )

        num_budgets = self._check_budget(
            num_budgets, max_perturbations=self.num_edges)

        self.num_budgets = num_budgets
        self.structure_attack = structure_attack
        self.feature_attack = feature_attack

        self.is_reseted = False

        return self
